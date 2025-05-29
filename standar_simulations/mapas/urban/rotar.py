#!/usr/bin/env python3
"""
#sumo -c osm.sumocfg --fcd-output trace.xml
#python3 /opt/sumo/tools/traceExporter.py -i trace.xml --ns2mobility-output=mobility.tcl
"""
import math
import xml.etree.ElementTree as ET
import argparse
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Rotar un mapa .osm alrededor de su centróide.")
    parser.add_argument('-i', '--input', required=True, help='Archivo OSM de entrada')
    parser.add_argument('-o', '--output', required=True, help='Archivo OSM de salida')
    parser.add_argument('-a', '--angle', type=float, default=-12.0,
                        help='Ángulo de rotación en grados (positivo = antihorario)')
    return parser.parse_args()

def compute_centroid(nodes):
    lats = [float(node.attrib['lat']) for node in nodes]
    lons = [float(node.attrib['lon']) for node in nodes]
    if not lats or not lons:
        raise ValueError("El archivo OSM no contiene nodos con coordenadas.")
    return sum(lats) / len(lats), sum(lons) / len(lons)

def rotate_point(lat, lon, angle_rad, origin_lat, origin_lon):
    # Convertir lat/lon a coordenadas cartesianas simples
    x = (lon - origin_lon) * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat)
    # Aplicar rotación
    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    # Convertir de nuevo a lat/lon
    lat_rot = y_rot + origin_lat
    lon_rot = x_rot / math.cos(math.radians(origin_lat)) + origin_lon
    return lat_rot, lon_rot

if __name__ == '__main__':
    args = parse_args()

    try:
        tree = ET.parse(args.input)
    except Exception as e:
        print(f"ERROR: No se pudo leer el archivo '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    root = tree.getroot()
    # Manejar posible namespace
    m = re.match(r"\{.*\}", root.tag)
    ns = m.group(0) if m else ''

    nodes = root.findall(f'{ns}node')
    if not nodes:
        print("ERROR: No se encontraron nodos en el archivo OSM.", file=sys.stderr)
        sys.exit(1)

    # Calcular centróide y ángulo en radianes
    centroid_lat, centroid_lon = compute_centroid(nodes)
    angle_rad = math.radians(args.angle)

    # Rotar cada nodo
    for node in nodes:
        lat = float(node.attrib['lat'])
        lon = float(node.attrib['lon'])
        lat_r, lon_r = rotate_point(lat, lon, angle_rad, centroid_lat, centroid_lon)
        node.set('lat', f"{lat_r:.7f}")
        node.set('lon', f"{lon_r:.7f}")

    # Guardar resultado
    try:
        tree.write(args.output, encoding='utf-8', xml_declaration=True)
        print(f"Archivo rotado guardado en '{args.output}' (rotación de {args.angle}°).")
    except Exception as e:
        print(f"ERROR: No se pudo escribir el archivo '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)
