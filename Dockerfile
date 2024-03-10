FROM python:3.11

RUN apt-get update
RUN apt-get install -y vim less wget locales openjdk-17-jre && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV LANG='ja_JP.UTF-8'
ENV LANGUAGE='ja_JP:ja'
ENV LC_ALL='ja_JP.UTF-8'
ENV TZ JST-9
ENV TERM xterm

ARG project_dir=/projects/
WORKDIR $project_dir

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

# ビルド時にライブラリをインストールするとき
COPY ./requirements.txt .
RUN pip install -r requirements.txt
