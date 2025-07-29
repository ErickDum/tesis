#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"
#include "ns3/config-store.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/log.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/stats-module.h"
#include "ns3/mobility-building-info.h"
#include "ns3/buildings-helper.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/hybrid-buildings-propagation-loss-model.h"
#include "ns3/buildings-module.h"

#include <iomanip>
#include <ctime>


using namespace ns3;


NS_LOG_COMPONENT_DEFINE("v2x");

// *********************************************** FUNCION PRINCIPAL ***********************************************
int
main(int argc, char* argv[])
{   
    uint32_t now = static_cast<uint32_t> (std::time (nullptr));
    std::cout << now << std::endl;
    RngSeedManager::SetSeed (now);
    RngSeedManager::SetRun  (now % 1000);

    // ************************* PARÁMETROS DE SIMULACIÓN *************************
    // Paramatros de la topología

    // Parámetros de trafico
    uint32_t udpPacketSizeBe = 500;                         // 50 - 6000 bytes
    double dataRateBe = 1000;                               // kbps  de 2 a 50 mensajes por segundo
    uint32_t udpPacketSizeInt = 500;
    double dataRateInt = 1000;                              // kbps <= 65Mbps
    bool harqEnabled = true;
    Time delayBudget = Seconds(0);                          // Usa ventana seleccion T2
    Time delayBudget2 = Seconds(0);

    // Parámetros de la simulación
    Time simTime = Seconds(35);
    Time slBearersActivationTime = Seconds(20);

    // Parametros de NR
    uint16_t numerologyBwpSl = 1;
    double centralFrequencyBandSl = 5.9e9;                 // Banda n47 TDD
    uint16_t bandwidthBandSl = 1000;                       // 2000*100kHz = 100 MHz
    double txPower = 23;                                   // dBm
    //double txIntPower = 8;                               // dBm

    // Parametros resource allocation
    uint16_t sensingMethod = 1; 
    uint16_t resourceAllocationMethod = 1;

    std::string simTag = "default";
    std::string outputDir = "/home/tesis/Escritorio/";
    std::string mobility_file = "/home/tesis/Documentos/rl_training/ns-3-dev/scratch/mobility.tcl";
    // ****************************************************************************

    // ********************************** ARGUMENTOS ******************************
    int simTime_int = 35; 
    int slBearersActivationTime_int = 20;
    uint16_t ueNum = 2;
    uint16_t IntNum = 10; //682
    CommandLine cmd(__FILE__);
    cmd.AddValue("ueNum", "Number of UEs", ueNum);
    cmd.AddValue("IntNum", "Number of Interferers", IntNum);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime_int);
    cmd.AddValue("slBearerActivationTime",
                 "Sidelink bearer activation time in seconds",
                 slBearersActivationTime_int);
    cmd.AddValue("simTag", "Simulation tag for output files", simTag);
    cmd.AddValue("outputDir", "Output directory for simulation results", outputDir);
    cmd.AddValue("sensingMethod",
                 "Método de sensado: 0 antiguo, 1 nuevo",
                 sensingMethod);
    cmd.AddValue("resourceAllocationMethod",
                 "Método de asignación de recursos: 0 totalmente aleatorio, 1 aleatorio con rsrp, 2 greedy, 3 proportional fair ",
                 resourceAllocationMethod);
    cmd.AddValue("mobility_file", "Mobility file path", mobility_file);
    cmd.Parse(argc, argv);

    simTime = Seconds(simTime_int);
    slBearersActivationTime = Seconds(slBearersActivationTime_int);
    // ****************************************************************************

    Time finalSlBearersActivationTime = slBearersActivationTime + Seconds(0.01);
    Time finalSimTime = simTime + finalSlBearersActivationTime;
    Time finalSlBearersActivationTime2 = finalSlBearersActivationTime + Seconds(0.5);
    std::cout << "Duración de la simulacion: " << finalSimTime.GetSeconds() << std::endl;

    NS_ABORT_IF(centralFrequencyBandSl > 6e9);

    // Configuracion del tamaño del buffer
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(999999999));

     // ************************* CREACIÓN DE LA TOPOLOGÍA *************************
    NodeContainer ueNodesContainer;
    
    ueNodesContainer.Create(ueNum+IntNum);
    
    
    Ns2MobilityHelper sumo_trace (mobility_file);

    sumo_trace.Install();
    
    ///*
    // ***** Agregar edificios *****
    BuildingContainer buildings;
    Ptr<Building> building1 = CreateObject<Building>();
    building1->SetBoundaries(Box(938, 1032, 542, 638, 0.0, 15.0));
    building1->SetBuildingType(Building::Residential);
    building1->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building1);

    Ptr<Building> building2 = CreateObject<Building>();
    building2->SetBoundaries(Box(1048, 1132, 540, 630, 0.0, 15.0));
    building2->SetBuildingType(Building::Residential);
    building2->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building2);

    Ptr<Building> building3 = CreateObject<Building>();
    building3->SetBoundaries(Box(1148, 1244, 534, 627, 0.0, 15.0));
    building3->SetBuildingType(Building::Residential);
    building3->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building3);

    Ptr<Building> building4 = CreateObject<Building>();
    building4->SetBoundaries(Box(939, 1035, 435, 524, 0.0, 15.0));
    building4->SetBuildingType(Building::Residential);
    building4->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building4);

    Ptr<Building> building5 = CreateObject<Building>();
    building5->SetBoundaries(Box(1149, 1243, 424, 516, 0.0, 15.0));
    building5->SetBuildingType(Building::Residential);
    building5->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building5);

    Ptr<Building> building6 = CreateObject<Building>();
    building6->SetBoundaries(Box(942, 1037, 324, 417, 0.0, 15.0));
    building6->SetBuildingType(Building::Residential);
    building6->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building6);

    Ptr<Building> building7 = CreateObject<Building>();
    building7->SetBoundaries(Box(1050, 1102, 316, 409, 0.0, 15.0));
    building7->SetBuildingType(Building::Residential);
    building7->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building7);

    Ptr<Building> building8 = CreateObject<Building>();
    building8->SetBoundaries(Box(1157, 1252, 312, 409, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building8);

    Ptr<Building> building9 = CreateObject<Building>();
    building8->SetBoundaries(Box(1114, 1137, 372, 410, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building9);

    Ptr<Building> building10 = CreateObject<Building>();
    building8->SetBoundaries(Box(1112, 1140, 314, 350, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building10);

    Ptr<Building> building11 = CreateObject<Building>();
    building8->SetBoundaries(Box(1257, 1295, 528, 634, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building11);

    Ptr<Building> building12 = CreateObject<Building>();
    building8->SetBoundaries(Box(1262, 1287, 420, 524, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building12);

    Ptr<Building> building13 = CreateObject<Building>();
    building8->SetBoundaries(Box(1265, 1287, 300, 347, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building13);

    Ptr<Building> building14 = CreateObject<Building>();
    building8->SetBoundaries(Box(1263, 1285, 357, 417, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building14);

    Ptr<Building> building15 = CreateObject<Building>();
    building8->SetBoundaries(Box(1157, 1262, 278, 296, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building15);

    Ptr<Building> building16 = CreateObject<Building>();
    building8->SetBoundaries(Box(1046, 1146, 278, 300, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building16);

    Ptr<Building> building17 = CreateObject<Building>();
    building8->SetBoundaries(Box(933, 1044, 280, 305, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building17);

    Ptr<Building> building18 = CreateObject<Building>();
    building8->SetBoundaries(Box(889, 926, 318, 430, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building18);

    Ptr<Building> building19 = CreateObject<Building>();
    building8->SetBoundaries(Box(883, 923, 435, 532, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building19);

    Ptr<Building> building20 = CreateObject<Building>();
    building8->SetBoundaries(Box(879, 920, 539, 651, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building20);

    Ptr<Building> building21 = CreateObject<Building>();
    building8->SetBoundaries(Box(930, 1035, 655, 684, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building21);

    Ptr<Building> building22 = CreateObject<Building>();
    building8->SetBoundaries(Box(1042, 1132, 649, 674, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building22);

    Ptr<Building> building23 = CreateObject<Building>();
    building8->SetBoundaries(Box(1141, 1248, 644, 673, 0.0, 15.0));
    building8->SetBuildingType(Building::Residential);
    building8->SetExtWallsType(Building::ConcreteWithWindows);
    buildings.Add(building23);

    BuildingsHelper::Install(ueNodesContainer);

    //*/
    // ****************************************************************************


    // ************************ CONFIGURACIÓN DE NR HELPER ************************
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetEpcHelper(epcHelper);
    /*
     * Division de espectro
     */
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    /* Creación de la configuración de la banda de operación
     */
    CcBwpCreator::SimpleOperationBandConf bandConfSl(centralFrequencyBandSl,
                                                     bandwidthBandSl,
                                                     numCcPerBand,
                                                     BandwidthPartInfo::V2V_Urban);

    OperationBandInfo bandSl = ccBwpCreator.CreateOperationBandContiguousCc(bandConfSl);


    /*
     * Configuración de la condición del canal y el modelo de propagación
     */
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(100)));
    nrHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));


    nrHelper->InitializeOperationBand(&bandSl);
    allBwps = CcBwpCreator::GetAllBwps({bandSl});

    Packet::EnableChecking();
    Packet::EnablePrinting();

    /*
     *  Case (i): Atributos válidos para todos los nodos
     */
    // Latencia del core
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    NetDeviceContainer uePlatooningNetDev;
    NetDeviceContainer ueIntNetDev;

    uint8_t bwpIdForGbrMcptt = 0;
    uint8_t bwpIdForVoice = 0;

    std::set<uint8_t> bwpIdContainer;
    bwpIdContainer.insert(bwpIdForGbrMcptt);
    bwpIdContainer.insert(bwpIdForVoice);

    for (uint32_t i = 0; i < ueNodesContainer.GetN(); i++)
    {
        Ptr<Node> node = ueNodesContainer.Get(i);
        if (i < ueNum)
        {   
            // Configuración de los nodos UE platooning
            nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
            nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
            nrHelper->SetUeAntennaAttribute("AntennaElement",
                                            PointerValue(CreateObject<IsotropicAntennaModel>()));

            nrHelper->SetUePhyAttribute("TxPower", DoubleValue(txPower));
            nrHelper->SetUeMacTypeId(NrSlUeMac::GetTypeId());
            nrHelper->SetUeMacAttribute("EnableSensing", BooleanValue(true));
            nrHelper->SetUeMacAttribute("SensingMethod", UintegerValue(sensingMethod));     //Método de sensado: 0 antiguo, 1 nuevo
            //Método de asignación de recursos: 0 totalmente aleatorio, 1 aleatorio con rsrp, 2 greedy, 3 proportional fair 
            nrHelper->SetUeMacAttribute("ResourceAllocationMethod", UintegerValue(resourceAllocationMethod));
            nrHelper->SetUeMacAttribute("T1", UintegerValue(2));
            nrHelper->SetUeMacAttribute("T2", UintegerValue(10));
            nrHelper->SetUeMacAttribute("ActivePoolId", UintegerValue(0));

            nrHelper->SetBwpManagerTypeId(TypeId::LookupByName("ns3::NrSlBwpManagerUe"));
            nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_V2X", UintegerValue(bwpIdForGbrMcptt));
            nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_V2X", UintegerValue(bwpIdForVoice)); 
            NetDeviceContainer singleDev = nrHelper->InstallUeDevice(node, allBwps);
            uePlatooningNetDev.Add(singleDev);
        }
        else
        {
            // Configuración de los nodos UE interferentes
            nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
            nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
            nrHelper->SetUeAntennaAttribute("AntennaElement",
                                            PointerValue(CreateObject<IsotropicAntennaModel>()));

            nrHelper->SetUePhyAttribute("TxPower", DoubleValue(txPower));

            // Si es posible, ajustar T1/T2 para interferentes (ejemplo)
            nrHelper->SetUeMacAttribute("EnableSensing", BooleanValue(false));
            nrHelper->SetUeMacAttribute("ResourceAllocationMethod", UintegerValue(1));
            nrHelper->SetUeMacAttribute("T1", UintegerValue(2)); // Valor distinto
            nrHelper->SetUeMacAttribute("T2", UintegerValue(10));
            nrHelper->SetUeMacAttribute("ActivePoolId", UintegerValue(0));
            
            NetDeviceContainer singleDev = nrHelper->InstallUeDevice(node, allBwps);
            ueIntNetDev.Add(singleDev);

        }
    }
    // ****************************************************************************

    // ************************* CONFIGURACIÓN DE SIDELINK ************************
    Ptr<NrSlHelper> nrSlHelper = CreateObject<NrSlHelper>();
    Ptr<NrSlHelper> IntSlHelper = CreateObject<NrSlHelper>();
    nrSlHelper->SetEpcHelper(epcHelper);
    IntSlHelper->SetEpcHelper(epcHelper);

    // Modelo de error
    std::string errorModel = "ns3::NrEesmIrT1";
    nrSlHelper->SetSlErrorModel(errorModel);
    IntSlHelper->SetSlErrorModel(errorModel);
    nrSlHelper->SetUeSlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    IntSlHelper->SetUeSlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));

    // Atributos del scheduler SL
    nrSlHelper->SetNrSlSchedulerTypeId(NrSlUeMacSchedulerFixedMcs::GetTypeId());
    nrSlHelper->SetUeSlSchedulerAttribute("Mcs", UintegerValue(14));
    IntSlHelper->SetNrSlSchedulerTypeId(NrSlUeMacSchedulerFixedMcs::GetTypeId());
    IntSlHelper->SetUeSlSchedulerAttribute("Mcs", UintegerValue(14));



    for (auto it = uePlatooningNetDev.Begin(); it != uePlatooningNetDev.End(); ++it)
    {
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();
    }
    for (auto it = ueIntNetDev.Begin(); it != ueIntNetDev.End(); ++it)
    {
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();
    }
    nrSlHelper->PrepareUeForSidelink(uePlatooningNetDev, bwpIdContainer);

    
    IntSlHelper->PrepareUeForSidelink(ueIntNetDev, bwpIdContainer);

    //********************************************************************************
    //************************ CONFIGURACION POOL SL USUARIOS ************************
    // SlResourcePoolNr
    LteRrcSap::SlResourcePoolNr slResourcePoolNr;
    Ptr<NrSlCommResourcePoolFactory> ptrFactory = Create<NrSlCommResourcePoolFactory>();
    
    // Configuración del pool
    std::vector<std::bitset<1>> slBitmap = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    ptrFactory->SetSlTimeResources(slBitmap);
    ptrFactory->SetSlSensingWindow(100); // T0 en ms
    ptrFactory->SetSlSelectionWindow(5);
    ptrFactory->SetSlFreqResourcePscch(10); // PSCCH RBs
    ptrFactory->SetSlSubchannelSize(25);
    ptrFactory->SetSlSubchannelSize(25);
    std::list<uint16_t> resourceReservePeriodList = {0, 100}; // ms
    ptrFactory->SetSlResourceReservePeriodList(resourceReservePeriodList);
    // Crear el pool
    LteRrcSap::SlResourcePoolNr pool = ptrFactory->CreatePool();
    slResourcePoolNr = pool;

    // Configurar el SlResourcePoolConfigNr IE
    LteRrcSap::SlResourcePoolConfigNr slresoPoolConfigNr;
    slresoPoolConfigNr.haveSlResourcePoolConfigNr = true;
    // ID del pool
    uint16_t poolId = 0;
    LteRrcSap::SlResourcePoolIdNr slResourcePoolIdNr;
    slResourcePoolIdNr.id = poolId;
    slresoPoolConfigNr.slResourcePoolId = slResourcePoolIdNr;
    slresoPoolConfigNr.slResourcePool = slResourcePoolNr;

    // Configurar el BWP IE
    LteRrcSap::SlBwpPoolConfigCommonNr slBwpPoolConfigCommonNr;
    // Array para los pools, insertamos el pool en el array según su poolId
    slBwpPoolConfigCommonNr.slTxPoolSelectedNormal[slResourcePoolIdNr.id] = slresoPoolConfigNr;

    // Configurar el BWP IE
    LteRrcSap::Bwp bwp;
    bwp.numerology = numerologyBwpSl;
    bwp.symbolsPerSlots = 14;
    bwp.rbPerRbg = 1;
    bwp.bandwidth = bandwidthBandSl;

    // Configurar el SlBwpGeneric IE
    LteRrcSap::SlBwpGeneric slBwpGeneric;
    slBwpGeneric.bwp = bwp;
    slBwpGeneric.slLengthSymbols = LteRrcSap::GetSlLengthSymbolsEnum(14);
    slBwpGeneric.slStartSymbol = LteRrcSap::GetSlStartSymbolEnum(0);

    // Configurar el SlBwpConfigCommonNr IE
    LteRrcSap::SlBwpConfigCommonNr slBwpConfigCommonNr;
    slBwpConfigCommonNr.haveSlBwpGeneric = true;
    slBwpConfigCommonNr.slBwpGeneric = slBwpGeneric;
    slBwpConfigCommonNr.haveSlBwpPoolConfigCommonNr = true;
    slBwpConfigCommonNr.slBwpPoolConfigCommonNr = slBwpPoolConfigCommonNr;

    // Configurar el SlFreqConfigCommonNr IE
    LteRrcSap::SlFreqConfigCommonNr slFreConfigCommonNr;
    // Insertar el BWP en la lista de BWP
    for (const auto& it : bwpIdContainer)
    {
        // it es el ID del BWP
        slFreConfigCommonNr.slBwpList[it] = slBwpConfigCommonNr;
    }

    // Configurar el TddUlDlConfigCommon IE
    LteRrcSap::TddUlDlConfigCommon tddUlDlConfigCommon;
    tddUlDlConfigCommon.tddPattern = "DL|DL|DL|F|UL|UL|UL|UL|UL|UL|";

    // Configurar el SlPreconfigGeneralNr IE
    LteRrcSap::SlPreconfigGeneralNr slPreconfigGeneralNr;
    slPreconfigGeneralNr.slTddConfig = tddUlDlConfigCommon;

    // Configurar el SlUeSelectedConfig IE
    LteRrcSap::SlUeSelectedConfig slUeSelectedPreConfig;
    slUeSelectedPreConfig.slProbResourceKeep = 1;
    // Configurar el SlPsschTxParameters IE
    LteRrcSap::SlPsschTxParameters psschParams;
    psschParams.slMaxTxTransNumPssch = 5;
    // Configurar el SlPsschTxConfigList IE
    LteRrcSap::SlPsschTxConfigList pscchTxConfigList;
    pscchTxConfigList.slPsschTxParameters[0] = psschParams;
    slUeSelectedPreConfig.slPsschTxConfigList = pscchTxConfigList;

    // Configurar el SlPreconfigNr IE
    LteRrcSap::SidelinkPreconfigNr slPreConfigNr;
    slPreConfigNr.slPreconfigGeneral = slPreconfigGeneralNr;
    slPreConfigNr.slUeSelectedPreConfig = slUeSelectedPreConfig;
    slPreConfigNr.slPreconfigFreqInfoList[0] = slFreConfigCommonNr;

    // Instalar la configuración previa de NR SL
    nrSlHelper->InstallNrSlPreConfiguration(uePlatooningNetDev, slPreConfigNr);


    //*******************************************************************************
    //********************* CONFIGURACION POOL SL INTERFERENTES *********************
    // SlResourcePoolNr
    LteRrcSap::SlResourcePoolNr slResourcePoolInt;
    Ptr<NrSlCommResourcePoolFactory> ptrFactoryInt = Create<NrSlCommResourcePoolFactory>();
    
    
    // Configuración del pool
    std::vector<std::bitset<1>> slBitmapInt = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    ptrFactoryInt->SetSlTimeResources(slBitmapInt);
    ptrFactoryInt->SetSlSensingWindow(100); // T0 en ms
    ptrFactoryInt->SetSlSelectionWindow(10);
    ptrFactoryInt->SetSlFreqResourcePscch(10); // PSCCH RBs
    ptrFactoryInt->SetSlSubchannelSize(25);
    ptrFactoryInt->SetSlMaxNumPerReserve(3);
    std::list<uint16_t> resourceReservePeriodListInt = {0, 100}; // ms
    ptrFactoryInt->SetSlResourceReservePeriodList(resourceReservePeriodListInt);
    // Crear el pool
    LteRrcSap::SlResourcePoolNr poolInt = ptrFactoryInt->CreatePool();
    slResourcePoolInt = poolInt;

    // Configurar el SlResourcePoolConfigNr IE
    LteRrcSap::SlResourcePoolConfigNr slresoPoolConfigInt;
    slresoPoolConfigInt.haveSlResourcePoolConfigNr = true;
    // ID del pool
    uint16_t poolIdInt = 0;
    LteRrcSap::SlResourcePoolIdNr slResourcePoolIdInt;
    slResourcePoolIdInt.id = poolIdInt;
    slresoPoolConfigInt.slResourcePoolId = slResourcePoolIdInt;
    slresoPoolConfigInt.slResourcePool = slResourcePoolInt;

    // Configurar el BWP IE
    LteRrcSap::SlBwpPoolConfigCommonNr slBwpPoolConfigCommonInt;
    // Array para los pools, insertamos el pool en el array según su poolId
    slBwpPoolConfigCommonInt.slTxPoolSelectedNormal[slResourcePoolIdNr.id] = slresoPoolConfigInt;

    // Configurar el BWP IE
    LteRrcSap::Bwp bwpInt;
    bwpInt.numerology = numerologyBwpSl;
    bwpInt.symbolsPerSlots = 14;
    bwpInt.rbPerRbg = 1;
    bwpInt.bandwidth = bandwidthBandSl;

    // Configurar el SlBwpGeneric IE
    LteRrcSap::SlBwpGeneric slBwpGenericInt;
    slBwpGenericInt.bwp = bwpInt;
    slBwpGenericInt.slLengthSymbols = LteRrcSap::GetSlLengthSymbolsEnum(14);
    slBwpGenericInt.slStartSymbol = LteRrcSap::GetSlStartSymbolEnum(0);

    // Configurar el SlBwpConfigCommonNr IE
    LteRrcSap::SlBwpConfigCommonNr slBwpConfigCommonInt;
    slBwpConfigCommonInt.haveSlBwpGeneric = true;
    slBwpConfigCommonInt.slBwpGeneric = slBwpGenericInt;
    slBwpConfigCommonInt.haveSlBwpPoolConfigCommonNr = true;
    slBwpConfigCommonInt.slBwpPoolConfigCommonNr = slBwpPoolConfigCommonInt;

    // Configurar el SlFreqConfigCommonNr IE
    LteRrcSap::SlFreqConfigCommonNr slFreConfigCommonInt;
    // Insertar el BWP en la lista de BWP
    for (const auto& it : bwpIdContainer)
    {
        // it es el ID del BWP
        slFreConfigCommonInt.slBwpList[it] = slBwpConfigCommonInt;
    }

    // Configurar el TddUlDlConfigCommon IE
    LteRrcSap::TddUlDlConfigCommon tddUlDlConfigCommonInt;
    tddUlDlConfigCommonInt.tddPattern = "DL|DL|DL|F|UL|UL|UL|UL|UL|UL|";

    // Configurar el SlPreconfigGeneralNr IE
    LteRrcSap::SlPreconfigGeneralNr slPreconfigGeneralInt;
    slPreconfigGeneralInt.slTddConfig = tddUlDlConfigCommonInt;

    // Configurar el SlUeSelectedConfig IE
    LteRrcSap::SlUeSelectedConfig slUeSelectedPreConfigInt;
    slUeSelectedPreConfigInt.slProbResourceKeep = 1;
    // Configurar el SlPsschTxParameters IE
    LteRrcSap::SlPsschTxParameters psschParamsInt;
    psschParamsInt.slMaxTxTransNumPssch = 5;
    // Configurar el SlPsschTxConfigList IE
    LteRrcSap::SlPsschTxConfigList pscchTxConfigListInt;
    pscchTxConfigListInt.slPsschTxParameters[0] = psschParamsInt;
    slUeSelectedPreConfigInt.slPsschTxConfigList = pscchTxConfigListInt;

    // Configurar el SlPreconfigNr IE
    LteRrcSap::SidelinkPreconfigNr slPreConfigInt;
    slPreConfigInt.slPreconfigGeneral = slPreconfigGeneralInt;
    slPreConfigInt.slUeSelectedPreConfig = slUeSelectedPreConfigInt;
    slPreConfigInt.slPreconfigFreqInfoList[0] = slFreConfigCommonInt;



    IntSlHelper->InstallNrSlPreConfiguration(ueIntNetDev, slPreConfigInt);
    // ****************************************************************************
    // ************************* CONFIGURACIÓN DE INTERNET *************************
    int64_t stream = 1;
    stream += nrHelper->AssignStreams(uePlatooningNetDev, stream);
    stream += nrSlHelper->AssignStreams(uePlatooningNetDev, stream);
    stream += nrHelper->AssignStreams(ueIntNetDev, stream);
    stream += IntSlHelper->AssignStreams(ueIntNetDev, stream);

    /*
     * Configurar el stack de IP
     */

    InternetStackHelper internet;
    internet.Install(ueNodesContainer);
    stream += internet.AssignStreams(ueNodesContainer, stream);
    //internet.Install(ueVoiceContainer);
    //stream += internet.AssignStreams(ueVoiceContainer, stream);
    //internet.Install(ueIntContainer);
    //stream2 += internet.AssignStreams(ueIntContainer, stream2);

    uint32_t dstL2Id = 255;
    uint32_t dstL2Id2 = 254;
    Ipv4Address groupAddress4("225.0.0.0"); // Usar dirección multicast como destino
    Address remoteAddress;
    Address localAddress;
    uint16_t port = 8000;
    Ptr<LteSlTft> tft;
    SidelinkInfo slInfo;
    slInfo.m_castType = SidelinkInfo::CastType::Unicast;
    slInfo.m_dstL2Id = dstL2Id;
    slInfo.m_rri = MilliSeconds(100);
    slInfo.m_pdb = delayBudget;
    slInfo.m_harqEnabled = harqEnabled;


    Ipv4Address groupAddress42("10.10.10.10"); // Usar dirección multicast como destino
    Address remoteAddress2;
    Address localAddress2;
    uint16_t port2 = 8000;
    Ptr<LteSlTft> tft2;
    SidelinkInfo slInfo2;
    slInfo2.m_castType = SidelinkInfo::CastType::Groupcast;
    slInfo2.m_dstL2Id = dstL2Id2;
    slInfo2.m_rri = MilliSeconds(100);
    slInfo2.m_pdb = delayBudget;
    slInfo2.m_harqEnabled = harqEnabled;

    
    // Assign IP addresses to platooning UEs
    Ipv4InterfaceContainer uePlatooningIpIface;
    uePlatooningIpIface = epcHelper->AssignUeIpv4Address(uePlatooningNetDev);
    Ipv4InterfaceContainer ueIntIpIface;
    ueIntIpIface = epcHelper->AssignUeIpv4Address(ueIntNetDev);

    //PrintIpAddresses(ueNodesContainer, ueIntIpIface, "Nodos Int");

    // Configurar la puerta de enlace predeterminada para el UE
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < ueNodesContainer.GetN(); ++u)
    {
        Ptr<Node> ueNode = ueNodesContainer.Get(u);
        // Establecer la puerta de enlace predeterminada para el UE
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    remoteAddress = InetSocketAddress(groupAddress4, port);
    localAddress = InetSocketAddress(Ipv4Address::GetAny(), port);
    tft = Create<LteSlTft>(LteSlTft::Direction::BIDIRECTIONAL, groupAddress4, slInfo);
    // Establecer Sidelink bearers
    nrSlHelper->ActivateNrSlBearer(finalSlBearersActivationTime, uePlatooningNetDev, tft);

    remoteAddress2 = InetSocketAddress(groupAddress42, port2);
    localAddress2 = InetSocketAddress(Ipv4Address::GetAny(), port2);
    tft2 = Create<LteSlTft>(LteSlTft::Direction::BIDIRECTIONAL, groupAddress42, slInfo2);
    // Establecer Sidelink bearers
    IntSlHelper->ActivateNrSlBearer(finalSlBearersActivationTime, ueIntNetDev, tft2);

    /*
    * Configurar las aplicaciones
    */

    // Configurar la aplicación de voz en el transmisor (primer nodo)
    OnOffHelper sidelinkClient("ns3::UdpSocketFactory", remoteAddress);
    sidelinkClient.SetAttribute("EnableSeqTsSizeHeader", BooleanValue(true));
    std::string dataRateBeString = std::to_string(dataRateBe) + "kb/s";
    std::cout << "Data rate " << DataRate(dataRateBeString) << std::endl;
    sidelinkClient.SetConstantRate(DataRate(dataRateBeString), udpPacketSizeBe);
    ApplicationContainer clientApps = sidelinkClient.Install(ueNodesContainer.Get(0)); // Primer nodo como transmisor

    OnOffHelper sidelinkClient2("ns3::UdpSocketFactory", remoteAddress2);
    sidelinkClient2.SetAttribute("EnableSeqTsSizeHeader", BooleanValue(true));
    std::string dataRateIntString = std::to_string(dataRateInt) + "kb/s";
    sidelinkClient2.SetConstantRate(DataRate(dataRateIntString), udpPacketSizeInt);
    ApplicationContainer clientApps2;
    for (uint32_t u = ueNum; u < ueNodesContainer.GetN(); ++u)
    {
        Ptr<Node> ueNode = ueNodesContainer.Get(u);
        ApplicationContainer clientApps2ForNode = sidelinkClient2.Install(ueNode);
        clientApps2.Add(clientApps2ForNode);
    }

    // Iniciar y detener la aplicación de transmisión
    clientApps.Start(finalSlBearersActivationTime);
    clientApps.Stop(finalSimTime);
    clientApps2.Start(finalSlBearersActivationTime2);
    clientApps2.Stop(finalSimTime);

    // Calcular tiempos de inicio y finalización
    double realAppStart =
        finalSlBearersActivationTime.GetSeconds() +
        ((double)udpPacketSizeBe * 8.0 / (DataRate(dataRateBeString).GetBitRate()));
    double appStopTime = (finalSimTime).GetSeconds();

    std::cout << "App inicia en " << realAppStart << " sec" << std::endl;
    std::cout << "App se detiene en " << appStopTime << " sec" << std::endl;

    // Configurar las aplicaciones de recepción en todos los nodos excepto el transmisor
    ApplicationContainer serverApps;
    PacketSinkHelper sidelinkSink("ns3::UdpSocketFactory", localAddress);
    sidelinkSink.SetAttribute("EnableSeqTsSizeHeader", BooleanValue(true));

    for (uint32_t i = 1; i < ueNum; ++i) // Desde el nodo 1 hasta el último
    {
        ApplicationContainer app = sidelinkSink.Install(ueNodesContainer.Get(i));
        serverApps.Add(app);
    }

    std::cout << "Instaladas aplicaciones en nodos receptores" << std::endl;

    // Iniciar las aplicaciones de recepción
    serverApps.Start(Seconds(2.0));
    serverApps.Stop(finalSimTime);

    // *************************************************************************************

    // ****************************** INICIO DE LA SIMULACIÓN ******************************
    std::cout << "Starting simulation..." << std::endl;
    Simulator::Stop(finalSimTime);
    Simulator::Run();

    Simulator::Destroy();
    std::cout << "Simulation finished." << std::endl;
    // *************************************************************************************
    return 0;
}
// *****************************************************************************************************************