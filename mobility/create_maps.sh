#!/bin/bash

max_interferentes=33
step=1
sims_per_step=10
scenario="urban"  # Change this to the desired scenario (e.g., "urban", "highway".)
script_path="./sumo/$scenario/"

for ((i=1; i<max_interferentes; i+=step))
do
    for ((j=0; j<sims_per_step; j++))
    do 
        echo "Creating map with $i interferentes, simulation $j"

        # Crear rutas (solor urban)
        python3 /opt/sumo/tools/randomTrips.py \
                -v -b 0 -e 500 -p 1 \
                -n ./sumo/${scenario}/map_${scenario}.net.xml \
                -r ./sumo/${scenario}/map_${scenario}.rou.xml
        
        # Limpiar fichero temporal
        rm -f trips.trips.xml

        # Crear rutas de pelotón e interferentes
        python3 ./sumo/${scenario}/make_${scenario}_map.py "$i" "$script_path"

        # Ejecutar SUMO
        sumo -c ./sumo/${scenario}/map_${scenario}.sumocfg --fcd-output ./sumo/${scenario}/trace_${scenario}.xml

        # Exportar trace a formato NS2/Mobility
        python3 /opt/sumo/tools/traceExporter.py \
            -i ./sumo/${scenario}/trace_${scenario}.xml \
            --ns2mobility-output="./mapas/$scenario/mobility_int_${i}_sim_${j}.tcl" 
    done
done
