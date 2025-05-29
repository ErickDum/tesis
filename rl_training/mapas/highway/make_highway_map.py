# Simular de 60 a 130 segundos
import sys
import random

def get_random_routes(root):
    routes = set()
    for elem in root:
        for child in elem:
            routes.add(child.attrib['edges'])
    return routes

def gen_routes(vtype_defs, size_platoon, inft_num, output_file):
    print("Generando rutas aleatorias...")

    d_i = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

    i_d = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]

    if random.randint(0, 1) == 0:
        print("Ruta derecha a izquierda seleccionada")
        route_list = d_i
        alternative_route = i_d
    else:
        print("Ruta izquierda a derecha seleccionada")
        route_list = i_d
        alternative_route = d_i
    route = " ".join(route_list)

    remaining_interf = 0
    if inft_num  > 66:
        remaining_interf = inft_num - 66
        inft_num = 66

    front_interf = inft_num // 2
    back_interf = inft_num - front_interf

    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write("    <!-- Vehicle definitions -->\n")
        for vt in vtype_defs:
            f.write(f"    <vType {vt} />\n")

        f.write("    <!-- Platoon -->\n")
        id_v = 0
        time = 0.0

        for j in range(size_platoon):
            vtype = "leader" if j == 0 else "follower"
            f.write(f'    <vehicle id="{id_v}" type="{vtype}" depart="{time:.2f}" '
                    f'departLane="first" departSpeed="0">\n')
            f.write(f'        <route edges="{route}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1

        f.write("    <!-- Interferentes atras -->\n")
        for _ in range(back_interf):
            f.write(f'    <vehicle id="{id_v}" type="back_intf" depart="{time:.2f}" '
                    f'departLane="best" departSpeed="0">\n')
            f.write(f'        <route edges="{route}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1
        
        time = 20 - ((remaining_interf//12) * 1.25)
        
        f.write("    <!-- Interferentes restantes -->\n")
        while remaining_interf > 0:
            subroute = " ".join(alternative_route[4:])    
            f.write(f'    <vehicle id="{id_v}" type="intf" depart="{time:.2f}" '
                f'departLane="best" departSpeed="0">\n')
            f.write(f'        <route edges="{subroute}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1
            remaining_interf -= 1

            subroute = " ".join(alternative_route[3:])    
            f.write(f'    <vehicle id="{id_v}" type="intf" depart="{time:.2f}" '
                f'departLane="best" departSpeed="0">\n')
            f.write(f'        <route edges="{subroute}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1
            remaining_interf -= 1

            subroute = " ".join(alternative_route[2:])    
            f.write(f'    <vehicle id="{id_v}" type="intf" depart="{time:.2f}" '
                f'departLane="best" departSpeed="0">\n')
            f.write(f'        <route edges="{subroute}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1
            remaining_interf -= 1

            subroute = " ".join(alternative_route[1:])    
            f.write(f'    <vehicle id="{id_v}" type="intf" depart="{time:.2f}" '
                f'departLane="best" departSpeed="0">\n')
            f.write(f'        <route edges="{subroute}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1
            remaining_interf -= 1

        time = 45 - ((front_interf//3) * 1.25)

        f.write("    <!-- Interferentes adelante -->\n")
        for _ in range(front_interf):
            subroute = " ".join(route_list[2:])
            f.write(f'    <vehicle id="{id_v}" type="front_intf" depart="{time:.2f}" '
                    f'departLane="best" departSpeed="0">\n')
            f.write(f'        <route edges="{subroute}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1
        

        f.write('</routes>\n')

        print("Rutas generadas y guardadas en", output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python script.py  <INTERFERENTES>")
        sys.exit(1)

    SIZE_PLATOON = 4
    INTERFERENTES = int(sys.argv[1])
    OUTPUT_ROUTES = "/home/erick/Documentos/ns-3-dev/mapa/highway/platooning_highway.rou.xml"

    vtype_defs = [
        'id="leader" accel="3.0" decel="6.0" minGap="1" color="1,0,0" maxSpeed="19" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0"',

        'id="follower" accel="3.0" decel="6.0" minGap="0.5" color="0,0,1" maxSpeed="19.1" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0"',

        'id="back_intf" accel="3.0" decel="6.0" color="0,1,0" maxSpeed="19.1" '
        'speedFactor="1.0" speedDev="0.1"',

        'id="front_intf" accel="3.0" decel="6.0" minGap="1" color="0,1,0" maxSpeed="19" '
        'speedFactor="1.0" speedDev="0.1"',

        'id="intf" accel="3.0" decel="6.0" minGap="1" color="1,1,0" maxSpeed="15" '
        'speedFactor="1.0" speedDev="0.1"',
    ]


    gen_routes(vtype_defs, SIZE_PLATOON, INTERFERENTES, OUTPUT_ROUTES)
