FROM python:3.13-alpine

RUN pip3 install --upgrade pip==25.1.1
RUN pip3 install tensorboard

RUN apt-get update \
    && apt-get purge --auto-remove -yqq \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* \
    /usr/share/man \
    /usr/share/doc \
    /usr/share/doc-base

CMD ["tensorboard", "--logdir", "/logs/", "--host", "0.0.0.0", "--port", "6006"] 
