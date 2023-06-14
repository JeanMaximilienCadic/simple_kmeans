FROM python:3.10.12

RUN apt-get update -y && apt install  python3-tk -y
RUN groupadd -g 1000 ubuntu
RUN useradd -d /home/ubuntu -s /bin/bash -m ubuntu -u 1000 -g 1000
USER ubuntu
ENV HOME /home/ubuntu

COPY . /ubuntu/workspace
WORKDIR /ubuntu/workspace
RUN pip install -r requirements.txt

CMD ["python", "main.py"]