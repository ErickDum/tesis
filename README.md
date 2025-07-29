## **Proyecto:** Mejora de la Selección de Recursos para Comunicaciones Sidelink 5G

## Descripción

Este repositorio contiene la implementación, simulación y evaluación de un sistema de comunicación Dispositivo a Dispositivo (D2D) para entornos Beyond 5G (B5G) basado en redes cognitivas. El objetivo es mejorar el proceso estándar de selección de recursos en comunicaciones Sidelink de 5G mediante:

1. **Detector de energía** para filtrado de recursos.
2. **Asignación dinámica de recursos** usando Deep Q-Learning (DQL) y métodos heurísticos.

La validación se realizó en un escenario vehicular realista (pelotón) en el centro histórico de Cuenca, Ecuador, evaluando métricas de throughput, retraso, BLER y PRR.

## Estructura del Repositorio

```
├── mobility/
│   └── Archivos SUMO y scripts para generar movilidad de vehículos.
│
├── rl_training/
│   ├── ns3-modified/
│   │   └── Código NS-3 con integración de agentes DQL.
│   └── training_scripts/
│       └── Scripts Python y Bash para entrenar la red neuronal.
│
└── standard_simulations/
    ├── ns3-modified/
    │   └── Código NS-3 para simulaciones del estándar y métodos heurísticos (Greedy, Proportional Fair).
    └── simulation_scripts/
        └── Scripts de ejecución de simulaciones y recopilación de resultados.
```

## Requisitos Previos

* NS-3 con módulo 5G-LENA instalado
* SUMO (Simulation of Urban Mobility)
* Python 3.8+ con librerías: `tensorflow` / `pytorch` (según implementación), `numpy`, `pandas`, `matplotlib`

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/usuario/5g-sidelink-resource-allocation.git
   cd 5g-sidelink-resource-allocation
   ```
2. Instalar dependencias Python:

   ```bash
   pip install -r requirements.txt
   ```
3. Asegurarse de que NS-3 y SUMO estén accesibles desde la línea de comandos.


## Resultados

* **Filtrado por detector de energía:** mejora consistente en todas las métricas vs. estándar.
* **Métodos heurísticos:** pequeñas ganancias frente al esquema del estándar.
* **DQL (RL):** hasta +40% en throughput, -80% en retraso, +30% en BLER y +16% en PRR en alta densidad vehicular.

## Contribuciones

Las contribuciones principales incluyen:

* Diseño e implementación de un detector de energía para filtrado de recursos.
* Desarrollo de seis esquemas de asignación (2 heurísticos, 4 DQL).
* Setup de simulaciones integrando NS-3 (5G-LENA) con SUMO.
