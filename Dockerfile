FROM ubuntu:20.04 as builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata ca-certificates build-essential cmake make golang git sudo 

WORKDIR /build
COPY . .

RUN make

FROM ubuntu:20.04

WORKDIR /app

COPY --from=builder /usr/local/lib/libopencv*         /usr/local/lib/
COPY --from=builder /build/gin-tflite                 /app

ENTRYPOINT [ "./gin-tflite" ]