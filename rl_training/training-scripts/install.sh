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
sudo apt-get install sqlite sqlite3 libsqlite3-dev -y 
sudo apt-get install sqlite3 libsqlite3-dev -y
sudo apt-get install libeigen3-dev -y
sudo apt-get install gcc g++ python3 python3-pip cmake -y
sudo apt install python3.12-venv -y


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
cd ..

echo "=============================================================="
echo "================= INSTALACION DE NS3-GYM ====================="
echo "=============================================================="
sudo apt-get install libzmq5 libzmq5-dev -y
sudo apt-get install libprotobuf-dev -y
sudo apt-get install protobuf-compiler -y
cd contrib
git clone https://github.com/tkn-tub/ns3-gym
mv ./ns3-gym/ ./opengym 
cd ..
sed -i '207d' ./contrib/opengym/examples/linear-mesh-2/sim.cc
sed -i '352d' ./contrib/opengym/examples/linear-mesh/sim.cc


echo "=============================================================="
echo "=================== COMPILACION DE NS3 ======================="
echo "=============================================================="

#./ns3 configure --force-refresh -d optimized
./ns3 configure -d optimized
./ns3 build

echo "=============================================================="
echo "================= INSTALL PYTHON OPENGYM ====================="
echo "=============================================================="
python3 -m venv venv
source venv/bin/activate
cd ./contrib/opengym/ 
pip install -U ./model/ns3gym
pip install numpy
pip install torch
pip install gymnasium
pip install matplotlib
cd ../..

echo "=============================================================="
echo "================== INSTALACION COMPLETADA ===================="
echo "=============================================================="