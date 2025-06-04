#!/bin/bash

# Parametros de simulacion
interferentes_maximos=32
paso_interferentes=1
tamano_platoon=4
init_time=42
sim_time=58
path="/home/tesis/Documentos/simulaciones/"
mpath="/home/tesis/Documentos/mobility/mapas/urban/"
scenario="urban"


for ((sensingMethod=0; sensingMethod<=1; sensingMethod++))
do
    for ((i=0; i<=interferentes_maximos; i+=paso_interferentes))
    do
        for ((j=0; j<10; j++))
        do
            echo "Interferentes: $i - Simulación: $j"

            echo "Ejecutando simulación con $i interferentes..."
            ./ns3 run "scratch/v2x.cc --SimNum=$j --slBearerActivationTime=$init_time --simTime=$sim_time --ueNum=$tamano_platoon --IntNum=$i --mobility_file=${mpath}mobility_int_${i}_sim_${j}.tcl --simTag=urban_sensing_${sensingMethod}_allocation_1_interf_${i}_sim_${j} --outputDir=$path --sensingMethod=$sensingMethod --resourceAllocationMethod=1" 
                
        done
    done
    #Modificar ruta real y configurar rclone
    #rclone copy /home/tesis/Documentos/simulaciones/ gdrive:Simulaciones 
done



for ((resourceAllocationMethod=2; resourceAllocationMethod<=3; resourceAllocationMethod++))
do
    for ((i=0; i<=interferentes_maximos; i+=paso_interferentes))
    do
        for ((j=0; j<10; j++))
        do
            echo "Interferentes: $i - Simulación: $j"
            echo "Ejecutando simulación con $i interferentes..."
            ./ns3 run "scratch/v2x.cc --SimNum=$j --slBearerActivationTime=$init_time --simTime=$sim_time --ueNum=$tamano_platoon --IntNum=$i --mobility_file=${mpath}mobility_int_${i}_sim_${j}.tcl --simTag=urban_sensing_${1}_allocation_${resourceAllocationMethod}_interf_${i}_sim_${j} --outputDir=$path --sensingMethod=1 --resourceAllocationMethod=$resourceAllocationMethod" 
                
        done
    done
    #Modificar ruta real y configurar rclone
    #rclone copy /home/tesis/Documentos/simulaciones/ gdrive:Simulaciones 
done
