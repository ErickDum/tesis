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

# Número total de simulaciones
total_runs=10000

# Ejecutar simulaciones
for ((i = 1; i <= total_runs; i++)); do
    int_num=$(generate_triangular)
    echo "Run $i with IntNum=$int_num"
    ./ns3 run "scratch/v2x.cc --resourceAllocationMethod=4 --IntNum=$int_num"
done
