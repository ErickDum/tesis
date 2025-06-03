import xml.etree.ElementTree as ET
import sys
import random

def get_random_routes(root):
    routes = set()
    for elem in root:
        for child in elem:
            routes.add(child.attrib['edges'])
    return routes

def gen_routes(routes, vtype_defs, size_platoon, inft_num, output_file):
    print("Generando rutas aleatorias...")

    def distinct_count(route):
        # cuenta nodos distintos (ignorando sufijos "#...")
        return len({edge.split('#')[0] for edge in route.split()})

    # prefijos requeridos
    prefixes = {"E4", "E0", "E5", "E7", "E10", "E14", "E2"}
    # aristas especiales (sin sufijos)
    special = {
        "49217102", "542428845", "E12", "E9", "E8", "E15",
        "E3", "E18", "E19", "337277984", "1053072563", "E11"
    }

    # filtra rutas que empiecen con uno de los prefijos
    # y contengan al menos 2 aristas especiales
    filtered_routes = [
        r for r in routes
        if (r.split()[0].split('#')[0] in prefixes)
           and (sum(1 for e in r.split()
                    if e.split('#')[0] in special) >= 2)
    ]

    # ordena por cantidad de nodos distintos y toma las 6 más largas
    sorted_routes = sorted(filtered_routes, key=distinct_count, reverse=True)
    top_routes = sorted_routes[:6]

    # selecciona 1 ruta al azar de esas 6
    longest_route = random.choice(top_routes)
    route_list = longest_route.split()

    print("Ruta seleccionada:", len(route_list), "nodos distintos")

    remaining_interf = 0
    if inft_num  > 22:
        remaining_interf = inft_num - 22
        inft_num = 22

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
                    f'departLane="first" departSpeed="max">\n')
            f.write(f'        <route edges="{longest_route}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1

        f.write("    <!-- Interferentes atras -->\n")
        for _ in range(back_interf):
            f.write(f'    <vehicle id="{id_v}" type="back_intf" depart="{time:.2f}" '
                    f'departLane="first" departSpeed="max">\n')
            f.write(f'        <route edges="{longest_route}" />\n')
            f.write('    </vehicle>\n')
            id_v += 1

        time += 35 - (front_interf * 1.9)

        f.write("    <!-- Interferentes adelante -->\n")
        for _ in range(front_interf):
            subroute = " ".join(route_list[2:])
            f.write(f'    <vehicle id="{id_v}" type="front_intf" depart="{time:.2f}" '
                    f'departLane="first" departSpeed="max">\n')
            f.write(f'        <route edges="{subroute}" />\n')
            f.write('    </vehicle>\n')
            time += 1.75
            id_v += 1


        f.write("    <!-- Interferentes restantes -->\n")
        if remaining_interf > 0:
            # remaining interferers: pick routes with no edge in the platoon route_list
            candidates = [
                r for r in routes
                if all(edge not in route_list for edge in r.split())
            ]
            while remaining_interf > 0:
                r = random.choice(candidates)
                for _ in range(5):
                    if remaining_interf <= 0:
                        break
                    f.write(f'    <vehicle id="{id_v}" type="intf" depart="{time:.2f}" '
                        f'departLane="first" departSpeed="max">\n')
                    f.write(f'        <route edges="{r}" />\n')
                    f.write('    </vehicle>\n')
                    id_v += 1
                    remaining_interf -= 1

        f.write('</routes>\n')

        print("Rutas generadas y guardadas en", output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python script.py  <INTERFERENTES> <DIR>")
        sys.exit(1)

    SIZE_PLATOON = 4
    INTERFERENTES = int(sys.argv[1])
    DIR = sys.argv[2]
    INPUT_ROUTES = DIR + "/map_urban.rou.xml"
    OUTPUT_ROUTES = DIR + "/platooning_urban.rou.xml"

    vtype_defs = [
        'id="leader" accel="3.0" decel="6.0" minGap="1" color="1,0,0" maxSpeed="6" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0"',

        'id="follower" accel="3.0" decel="6.0" minGap="0.5" color="0,0,1" maxSpeed="6.1" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0"',

        'id="back_intf" accel="3.0" decel="6.0" color="0,1,0" maxSpeed="6.1" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0" '
        'speedFactor="1.0" speedDev="0.1"',

        'id="front_intf" accel="3.0" decel="6.0" minGap="1" color="0,1,0" maxSpeed="6" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0" ' 
        'speedFactor="1.0" speedDev="0.1"',

        'id="intf" accel="3.0" decel="6.0" minGap="1" color="1,1,0" maxSpeed="5" '
        'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0" '
        'speedFactor="1.0" speedDev="0.1"',
    ]

    tree = ET.parse(INPUT_ROUTES)
    root = tree.getroot()

    routes = get_random_routes(root)
    gen_routes(routes, vtype_defs, SIZE_PLATOON, INTERFERENTES, OUTPUT_ROUTES)
