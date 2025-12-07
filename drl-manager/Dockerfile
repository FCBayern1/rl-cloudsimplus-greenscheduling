FROM python:3.13-alpine

WORKDIR /mgr

RUN pip3 install --upgrade pip==25.1.1 && pip3 install stable-baselines3 sb3-contrib \
    tensorboard tqdm rich pyyaml pycurl scikit-learn

COPY drl-manager/gym_cloudsimplus gym_cloudsimplus

RUN pip3 install -e gym_cloudsimplus
RUN apt update \
    && apt purge --auto-remove -yqq \
    && apt autoremove -yqq --purge \
    && apt clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* \
    /usr/share/man \
    /usr/share/doc \
    /usr/share/doc-base