FROM ubuntu:latest

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y openssh-server clinfo openmpi-bin mpi mpich* make git pocl-opencl-icd ocl-icd-opencl-dev

RUN ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN mkdir /var/run/sshd

RUN echo 'root:root' |chpasswd

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN mkdir /root/.ssh

COPY id_rsa.pub /root/.ssh/authorized_keys

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]