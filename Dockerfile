FROM ubuntu:20.04 as builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata ca-certificates build-essential cmake make golang git sudo 

WORKDIR /build
COPY . .

RUN make

FROM ubuntu:20.04

WORKDIR /app

COPY --from=builder /usr/local/lib/libopencv*         /usr/local/lib/
COPY --from=builder /build/lib                        /app/lib
COPY --from=builder /build/models                     /app/models
COPY --from=builder /build/static                     /app/static
COPY --from=builder /build/gin-tflite                 /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates ffmpeg libgtk2.0-0 && apt clean

ENTRYPOINT [ "./gin-tflite" ]