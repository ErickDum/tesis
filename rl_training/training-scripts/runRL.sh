#!/bin/bash

# Función para generar un número con distribución triangular estándar (min=1, mode=15, max=30)
generate_triangular() {
    awk -v a=1 -v b=30 -v c=15 'BEGIN {
        u = rand();
        Fc = (c - a) / (b - a);
        if (u < Fc) {
            x = a + sqrt(u * (b - a) * (c - a));
        } else {
            x = b - sqrt((1 - u) * (b - a) * (b - c));
        }
        printf "%d\n", int(x + 0.5);  # Redondear al entero más cercano
    }'
}

# Parametros de simulacion
tamano_platoon=4
init_time=40
sim_time=60
path="/home/tesis/Documentos/simulaciones/"
scenario="urban"

# Número total de simulaciones
total_runs=10000

# Ejecutar simulaciones
for ((i = 1; i <= total_runs; i++)); do


    # Generación de trips aleatorios
    python3 /opt/sumo/tools/randomTrips.py \
        -v -b 0 -e 500 -p 1 \
        -n ./mapas/${scenario}/map_${scenario}.net.xml \
        -r ./mapas/${scenario}/map_${scenario}.rou.xml

    # Limpiar fichero temporal
    rm -f trips.trips.xml

    # Crear rutas de pelotón e interferentes
    python3 ./mapas/${scenario}/make_${scenario}_map.py "$i"

    # Ejecutar SUMO
    sumo -c ./mapas/${scenario}/map_${scenario}.sumocfg --fcd-output ./mapas/${scenario}/trace_${scenario}.xml

    # Exportar trace a formato NS2/Mobility
    python3 /opt/sumo/tools/traceExporter.py \
        -i ./mapas/${scenario}/trace_${scenario}.xml \
        --ns2mobility-output=./scratch/mobility.tcl


    int_num=$(generate_triangular)
    echo "Episodio $i con $int_num interferentes"
    #Poner en orden todos los paths de los diferenes ns3
    /home/carlos/Documents/rl_training/2/ns-3-dev/ns3 run "scratch/v2x.cc --slBearerActivationTime=$init_time --simTime=$sim_time --ueNum=$tamano_platoon --IntNum=$int_num --sensingMethod=1 --resourceAllocationMethod=4" &
    sleep 1
    ./ns3 run "scratch/v2x.cc --slBearerActivationTime=$init_time --simTime=$sim_time --ueNum=$tamano_platoon --IntNum=$int_num --sensingMethod=1 --resourceAllocationMethod=4" &
done
