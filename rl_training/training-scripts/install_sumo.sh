#!/bin/bash

# Este script sirve para instalar SUMO a partir de la construccion de archivos binarios
echo ==============================================================
echo ================ ACTUALIZACION DE PAQUETES ===================
echo ==============================================================
sudo apt update -y
sudo apt upgrade -y
sudo sudo apt install -y open-vm-tools-desktop

echo ==============================================================
echo =========== INSTALACION DE LIBRERIAS \(1\/6\) ===================
echo ==============================================================

sudo apt install -y git cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev python3-dev swig default-jdk maven libeigen3-dev


echo ==============================================================
echo ============ INSTALACION DE LIBRERIAS \(2\/6\) ==================
echo ==============================================================

sudo apt install -y ccache libavformat-dev libswscale-dev libopenscenegraph-dev python3-pip python3-setuptools

echo ==============================================================
echo ============= INSTALACION DE LIBRERIAS \(3\/6\) ==============
echo ==============================================================
sudo apt install -y libgtest-dev gettext tkdiff xvfb flake8 astyle python3-autopep8

echo ==============================================================
echo ============= INSTALACION DE LIBRERIAS \(4\/6\) =================
echo ==============================================================

sudo apt install -y python3-pandas python3-rtree python3-pyproj

echo ==============================================================
echo ============= INSTALACION DE LIBRERIAS \(5\/6\) =================
echo ==============================================================

sudo apt install -y python3-pyproj python3-rtree python3-pandas python3-flake8 python3-autopep8 python3-scipy python3-pulp python3-ezdxf

echo ==============================================================
echo ============== INSTALACION DE LIBRERIAS \(6\/6\) ================
echo ==============================================================

sudo apt install -y python3-pandas python3-rtree python3-pyproj

echo ==============================================================
echo =================== CAMBIO DE DIRECTORIO =====================
echo ==============================================================
cd /opt
echo "$PWD"

echo =================================================================
echo ========== DESCARGA SUMO DESDE EL REPOSITORIO  ===================
echo =================================================================
if test -d sumo; then
  echo "SUMO ya esta clonado."
else
  sudo git clone --recursive https://github.com/eclipse-sumo/sumo
  #sudo tar -zxvf sumo.tgz
fi

cd sumo
echo "$PWD"

echo =================================================================
echo ============ INSTALACION DE LIBRERIAS PYTHON ====================
echo =================================================================
sudo pip install -r tools/requirements.txt
sudo pip install jupyterlab


echo =================================================================
echo ============ CONFIGURA SUMO_HOME ================================
echo =================================================================
export SUMO_HOME="$PWD"
#sudo echo "export SUMO_HOME="$PWD"" >> ~/.bashrc
#####sudo echo "export SUMO_HOME="$PWD"" >> /etc/bash.bashrc
echo $SUMO_HOME

echo =================================================================
echo ============ INSTALACION DE SUMO ================================
echo =================================================================

sudo cmake -B build .
sudo cmake --build build -j$(nproc)
sudo cmake --install build
sudo echo export SUMO_HOME="/usr/local/share/sumo/" >> ~/.bashrc
source ~/.bashrc
echo "OK"
echo "Abra una nueva terminal y ejecute: sumo"
exit 0