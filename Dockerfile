FROM python:3.10-slim

LABEL author="Terézia Slanináková"
LABEL email="xslanin@mail.muni.cz"
LABEL website="https://disa.fi.muni.cz/complex-data-analysis/"

# Set the version of the image to use, default: cpu
ARG version=cpu

# Install linux packages
RUN apt-get update && apt-get install vim -y

# Install required python packages
COPY requirements-${version}.txt /tmp/
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements-${version}.txt

# Create user, make it the owner of the home directory
RUN addgroup --gid 1000 user && adduser --gid 1000 --uid 1000 --disabled-password --gecos user user
USER root
RUN chown -R user:user /home/user && chmod -R 755 /home/user

# Copy the files from the host to the container and install the local package
COPY . /home/user
RUN pip install -e /home/user

USER user
WORKDIR /home/user

CMD ['/bin/sh', '-c', 'bash']