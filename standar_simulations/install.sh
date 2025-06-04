#!/bin/bash

# Installation script for SUMO
chmod +x ./install_sumo.sh
./install_sumo.sh

# Update and upgrade packages
echo "=============================================================="
echo "================ ACTUALIZACION DE PAQUETES ==================="
echo "=============================================================="
sudo apt-get update -y

# Install dependencies
echo "=============================================================="
echo "============== INSTALACION DE LIBRERIAS NS3 =================="
echo "=============================================================="
sudo apt-get install libc6-dev -y
apt-get install sqlite sqlite3 libsqlite3-dev -y 
sudo apt-get install sqlite3 libsqlite3-dev -y
sudo apt-get install libeigen3-dev -y

# Install ns-3 5G LENA
echo "=============================================================="
echo "================== REPOSITORIO 5G LENA ======================="
echo "=============================================================="
git clone https://gitlab.com/cttc-lena/ns-3-dev.git
cd ns-3-dev
git checkout tags/ns-3-dev-v2x-v1.1 -b ns-3-dev-v2x-v1.1-branch
cd contrib
git clone https://gitlab.com/cttc-lena/nr.git
cd nr
git checkout tags/v2x-1.1 -b v2x-1.1-branch
cd ../..
cd ns-3-dev
./ns3 configure --enable-examples
./ns3 build