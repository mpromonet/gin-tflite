
importScripts('./mp4box.all.min.js');

const FRAME_BUFFER_TARGET_SIZE = 3;
const ENABLE_DEBUG_LOGGING = false;

function debugLog(msg) {
  if (!ENABLE_DEBUG_LOGGING)
    return;
  console.debug(msg);
}

class MP4PullDemuxer  {
  constructor(fileUri) {
    this.fileUri = fileUri;
  }

  async initialize() {
    this.source = new MP4Source(this.fileUri);
    this.readySamples = [];
    this._pending_read_resolver = null;

    await this._tracksReady();
    this._selectTrack(this.videoTrack);
  }

  getDecoderConfig() {
      return {
        // Browser doesn't support parsing full vp8 codec (eg: `vp08.00.41.08`),
        // they only support `vp8`.
        codec: this.videoTrack.codec.startsWith('vp08') ? 'vp8' : this.videoTrack.codec,
        displayWidth: this.videoTrack.track_width,
        displayHeight: this.videoTrack.track_height,
        description: this._getDescription(this.source.getDescriptionBox())
      }
  }

  async getNextChunk() {
    let sample = await this._readSample();
    const type = sample.is_sync ? "key" : "delta";
    const pts_us = (sample.cts * 1000000) / sample.timescale;
    const duration_us = (sample.duration * 1000000) / sample.timescale;
    return new EncodedVideoChunk({
      type: type,
      timestamp: pts_us,
      duration: duration_us,
      data: sample.data
    });
  }

  _getDescription(descriptionBox) {
    const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
    descriptionBox.write(stream);
    return new Uint8Array(stream.buffer, 8);  // Remove the box header.
  }

  async _tracksReady() {
    let info = await this.source.getInfo();
    this.videoTrack = info.videoTracks[0];
  }

  _selectTrack(track) {
    console.assert(!this.selectedTrack, "changing tracks is not implemented");
    this.selectedTrack = track;
    this.source.selectTrack(track);
  }

  async _readSample() {
    console.assert(this.selectedTrack);
    console.assert(!this._pending_read_resolver);

    if (this.readySamples.length) {
      return Promise.resolve(this.readySamples.shift());
    }

    let promise = new Promise((resolver) => { this._pending_read_resolver = resolver; });
    console.assert(this._pending_read_resolver);
    this.source.start(this._onSamples.bind(this));
    return promise;
  }

  _onSamples(samples) {
    const SAMPLE_BUFFER_TARGET_SIZE = 50;

    this.readySamples.push(...samples);
    if (this.readySamples.length >= SAMPLE_BUFFER_TARGET_SIZE)
      this.source.stop();

    let firstSampleTime = samples[0].cts * 1000000 / samples[0].timescale ;
    debugLog(`adding new ${samples.length} samples (first = ${firstSampleTime}). total = ${this.readySamples.length}`);

    if (this._pending_read_resolver) {
      this._pending_read_resolver(this.readySamples.shift());
      this._pending_read_resolver = null;
    }
  }
}

class MP4Source {
  constructor(uri) {
    this.file = MP4Box.createFile();
    this.file.onError = console.error.bind(console);
    this.file.onReady = this.onReady.bind(this);
    this.file.onSamples = this.onSamples.bind(this);

    debugLog('fetching file');
    fetch(uri).then(response => {
      debugLog('fetch responded');
      const reader = response.body.getReader();
      let offset = 0;
      let mp4File = this.file;

      function appendBuffers({done, value}) {
        if(done) {
          mp4File.flush();
          return;
        }
        let buf = value.buffer;
        buf.fileStart = offset;
        offset += buf.byteLength;
        mp4File.appendBuffer(buf);
        
        return reader.read().then(appendBuffers);
      }

      return reader.read().then(appendBuffers);
    })

    this.info = null;
    this._info_resolver = null;
  }

  onReady(info) {
    // TODO: Generate configuration changes.
    this.info = info;

    if (this._info_resolver) {
      this._info_resolver(info);
      this._info_resolver = null;
    }
  }

  getInfo() {
    if (this.info)
      return Promise.resolve(this.info);

    return new Promise((resolver) => { this._info_resolver = resolver; });
  }

  getDescriptionBox() {
    // TODO: make sure this is coming from the right track.
    const entry = this.file.moov.traks[0].mdia.minf.stbl.stsd.entries[0];
    const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
    if (!box) {
      throw new Error("avcC, hvcC, vpcC, or av1C box not found!");
    }
    return box;
  }

  selectTrack(track) {
    debugLog('selecting track %d', track.id);
    this.file.setExtractionOptions(track.id);
  }

  start(onSamples) {
    this._onSamples = onSamples;
    this.file.start();
  }

  stop() {
    this.file.stop();
  }

  onSamples(track_id, ref, samples) {
    this._onSamples(samples);
  }
}

class VideoRenderer {
  async initialize(demuxer, canvas) {
    this.frameBuffer = [];
    this.fillInProgress = false;

    this.demuxer = demuxer;
    await this.demuxer.initialize();
    const config = this.demuxer.getDecoderConfig();

    this.canvas = canvas;
    this.canvas.width = config.displayWidth;
    this.canvas.height = config.displayHeight;
    this.canvasCtx = canvas.getContext('2d');

    this.decoder = new VideoDecoder({
      output: this.bufferFrame.bind(this),
      error: e => console.error(e),
    });

    let support = await VideoDecoder.isConfigSupported(config);
    console.assert(support.supported);
    this.decoder.configure(config);

    this.init_resolver = null;
    let promise = new Promise((resolver) => this.init_resolver = resolver );

    this.fillFrameBuffer();
    return promise;
  }

  render(timestamp) {
    debugLog(`render(${timestamp})`);
    let frame = this.chooseFrame(timestamp);
    this.fillFrameBuffer();

    if (frame == null) {
      console.warn('VideoRenderer.render(): no frame ');
    } else {
      this.paint(frame);
    }
  }

  chooseFrame(timestamp) {
    if (this.frameBuffer.length == 0)
      return null;

    let minTimeDelta = Number.MAX_VALUE;
    let frameIndex = -1;

    for (let i = 0; i < this.frameBuffer.length; i++) {
      let time_delta = Math.abs(timestamp - this.frameBuffer[i].timestamp);
      if (time_delta < minTimeDelta) {
        minTimeDelta = time_delta;
        frameIndex = i;
      } else {
        break;
      }
    }

    console.assert(frameIndex != -1);

    if (frameIndex > 0)
      debugLog(`dropping ${frameIndex} stale frames`);

    for (let i = 0; i < frameIndex; i++) {
      let staleFrame = this.frameBuffer.shift();
      staleFrame.close();
    }

    let chosenFrame = this.frameBuffer[0];
    debugLog(`frame time delta = ${minTimeDelta/1000}ms (${timestamp} vs ${chosenFrame.timestamp})`)
    return chosenFrame;
  }

  async fillFrameBuffer() {
    if (this.frameBufferFull()) {
      debugLog('frame buffer full');

      if (this.init_resolver) {
        this.init_resolver();
        this.init_resolver = null;
      }

      return;
    }

    // This method can be called from multiple places and we some may already
    // be awaiting a demuxer read (only one read allowed at a time).
    if (this.fillInProgress) {
      return false;
    }
    this.fillInProgress = true;

    while (this.frameBuffer.length < FRAME_BUFFER_TARGET_SIZE &&
            this.decoder.decodeQueueSize < FRAME_BUFFER_TARGET_SIZE) {
      let chunk = await this.demuxer.getNextChunk();
      this.decoder.decode(chunk);
    }

    this.fillInProgress = false;

    // Give decoder a chance to work, see if we saturated the pipeline.
    setTimeout(this.fillFrameBuffer.bind(this), 0);
  }

  frameBufferFull() {
    return this.frameBuffer.length >= FRAME_BUFFER_TARGET_SIZE;
  }

  bufferFrame(frame) {
    debugLog(`bufferFrame(${frame.timestamp})`);
    this.frameBuffer.push(frame);
  }

  paint(frame) {
    this.canvasCtx.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);

    this.canvas.convertToBlob({"type":"image/jpeg"})
      .then(blob => fetch('/invoke/models/yolo/lite-model_yolo-v5-tflite_tflite_model_1.tflite', { method: 'POST', body: blob }))
      .then(r => r.json())
      .then(data => draw(data));    
  }
  draw(data) {
    const context = this.canvasCtx;
    context.strokeStyle = 'rgb(0,255,0)'
    context.fillStyle = 'rgb(0,255,0)'
    context.lineWidth = 2;
    context.font = '20px serif';

    if (data) {
      data.forEach(item => {
        context.fillText(item.ClassName, item.Box.Min.X, item.Box.Min.Y);
        context.rect(item.Box.Min.X, item.Box.Min.Y, (item.Box.Max.X - item.Box.Min.X), (item.Box.Max.Y - item.Box.Min.Y))
        context.stroke()
      })
    }  
  }
}

let playing = false
let lastMediaTimeSecs = 0;
let videoRenderer = new VideoRenderer();
let canvas = null;

self.addEventListener('message', async function(e) {
  console.info(`Worker message: ${JSON.stringify(e.data)}`);

  switch (e.data.command) {
    case 'initialize':
      canvas = e.data.canvas;
      postMessage({command: 'initialize-done'});
      break;
    case 'play':
      let videoDemuxer = new MP4PullDemuxer(e.data.videoFile);
      await videoRenderer.initialize(videoDemuxer, canvas);
      playing = true;
      lastMediaTimeSecs = performance.now();

      self.requestAnimationFrame(function renderVideo() {
        if (!playing)
          return;
        videoRenderer.render((performance.now() - lastMediaTimeSecs)*1000);
        self.requestAnimationFrame(renderVideo);
      });
      break;
    case 'stop':
      playing = false;
      break;
    default:
      console.error(`Worker bad message: ${e.data}`);
  }
});
