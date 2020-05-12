# ASP-OpenCL

El proyecto consta de 5 apartados
En cada carpeta se resuelve un problema de las practicas anteriores usando OpenCL.

Cada ejercicio tiene un makefile que tiene en cuenta el sistema operativo: Linux y MacOS

Los programas no necesitan ninguna manera especial de ejecutarse (salvo la version MPI mas adelante se explica).

Para poder compilarlos hacen falta las librerias MPI, OpenMP, OpenCL y un compilador de C.
Para instalarlos en ubuntu/debian:
```shell
sudo apt update && sudo apt install -y openmpi-bin mpi mpich* ocl-icd-opencl-dev
```

La version MPI combina las 3 tecnologias vistas en la asignatura

Cada ejecutable generado por los distintos makefile se encotrarán en su correspondiente carpeta. Ademas, podrán recibir parametros de entrada o no. Si no reciben nada se ejecutaran con un valor predefinido.

-   Add Numbers recibe un solo parametro `N` que representa los primeros `N` numeros que se han de sumar de forma paralela.
-   Igual pasa con el ejercicio de la convolucion. Recibe la cantidad de numeros que generar
-   El programa PI recibe el numero de simulaciones (de puntos) que se van a calcular.
-   La multiplicacion de matrices tiene como parametro la dimension de la matriz cuadrada que va a multiplicar con otra de igual dimension.

Ademas hemos generado una imagen de docker para linux que lleva todas las herramientas necesarias para la compilacion y ejecucion ademas de coger acceso a la GPU del host y arrancar un servidor ssh en el puerto 69.

Dentro de la carpeta `docker` se encuentra el Dockerfile y el script de creacion del contenedor.