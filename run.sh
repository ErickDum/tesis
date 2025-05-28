#!/bin/bash

# Parametros de simulacion
interferentes_maximos=33
paso_interferentes=2
tamano_platoon=4
init_time=40
sim_time=60
path="/home/tesis/Documentos/simulaciones/"
scenario="urban"

for ((sensingMethod=0; sensingMethod<=1; sensingMethod++))
do
    for ((i=0; i<=interferentes_maximos; i+=paso_interferentes))
    do
        for ((j=0; j<10; j++))
        do
            echo "Interferentes: $i - Simulación: $j"

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


            echo "Ejecutando simulación con $i interferentes..."
            # Correr simulación en ns-3 con tiempos enteros
            ./ns3 run "scratch/v2x.cc --slBearerActivationTime=$init_time --simTime=$sim_time --ueNum=$tamano_platoon --IntNum=$i --simTag=urban_sensing_${sensingMethod}_interf_${i}_sim_${j} --outputDir=$path --sensingMethod=$sensingMethod --resourceAllocationMethod=1" 
                
        done
    done
done
