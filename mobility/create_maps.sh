#!/bin/bash

max_interferentes=31
scenario="urban"  # Change this to the desired scenario (e.g., "urban", "highway".)
script_path="./sumo/$scenario/"
total=3000

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

for ((i = 1; i <=total; i++))
do
    echo "Starting simulation $i"
    int_num=$(generate_triangular)

    # Crear rutas (solor urban)
    python3 /opt/sumo/tools/randomTrips.py \
            -v -b 0 -e 500 -p 1 \
            -n ./sumo/${scenario}/map_${scenario}.net.xml \
            -r ./sumo/${scenario}/map_${scenario}.rou.xml
    # Limpiar fichero temporal
    rm -f trips.trips.xml

    # Crear rutas de pelotón e interferentes
    python3 ./sumo/${scenario}/make_${scenario}_map.py "$int_num" "$script_path"

    # Ejecutar SUMO
    sumo -c ./sumo/${scenario}/map_${scenario}.sumocfg --fcd-output ./sumo/${scenario}/trace_${scenario}.xml

    # Exportar trace a formato NS2/Mobility
    python3 /opt/sumo/tools/traceExporter.py \
        -i ./sumo/${scenario}/trace_${scenario}.xml \
        --ns2mobility-output="./mapas/mobility_${i}.tcl" 
    
    # Append the generated int_num to the summary file
    echo "$int_num" >> ./mapas/int_num.txt
done