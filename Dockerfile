FROM ubuntu:latest
LABEL authors="vovas"

ENTRYPOINT ["top", "-b"]