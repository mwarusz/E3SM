//===-- ocn/TracerDefs.inc --------------------*- C++ -*-===//
//
/// \file
/// \brief OMEGA Tracer Definition File
///
/// This file defines all OMEGA tracers. It is imported using
/// the C++ #include preprocessor directive in Tracers.cpp. Only the tracer
/// definitions selected in the Tracers section of the OMEGA YAML configuration
/// file will be allocated into memory.
//===----------------------------------------------------------------------===//

// Index defined for each tracer. The value of these tracer indices are still
// set in Tracers::init()  and tracers that have not been selected would have
// standard invalid value: Tracers::IndxInvalid = -1
inline static I4 IndxTemp   = Tracers::IndxInvalid;
inline static I4 IndxSalt   = Tracers::IndxInvalid;
inline static I4 IndxDebug1 = Tracers::IndxInvalid;
inline static I4 IndxDebug2 = Tracers::IndxInvalid;
inline static I4 IndxDebug3 = Tracers::IndxInvalid;
inline static I4 IndxDebug4 = Tracers::IndxInvalid;
inline static I4 IndxDebug5 = Tracers::IndxInvalid;
inline static I4 IndxDebug6 = Tracers::IndxInvalid;
inline static I4 IndxDebug7 = Tracers::IndxInvalid;
inline static I4 IndxDebug8 = Tracers::IndxInvalid;
inline static I4 IndxDebug9 = Tracers::IndxInvalid;
inline static I4 IndxDebug10 = Tracers::IndxInvalid;
inline static I4 IndxDebug11 = Tracers::IndxInvalid;
inline static I4 IndxDebug12 = Tracers::IndxInvalid;
inline static I4 IndxDebug13 = Tracers::IndxInvalid;
inline static I4 IndxDebug14 = Tracers::IndxInvalid;
inline static I4 IndxDebug15 = Tracers::IndxInvalid;
inline static I4 IndxDebug16 = Tracers::IndxInvalid;
inline static I4 IndxDebug17 = Tracers::IndxInvalid;
inline static I4 IndxDebug18 = Tracers::IndxInvalid;
inline static I4 IndxDebug19 = Tracers::IndxInvalid;
inline static I4 IndxDebug20 = Tracers::IndxInvalid;
inline static I4 IndxDebug21 = Tracers::IndxInvalid;
inline static I4 IndxDebug22 = Tracers::IndxInvalid;
inline static I4 IndxDebug23 = Tracers::IndxInvalid;
inline static I4 IndxDebug24 = Tracers::IndxInvalid;
inline static I4 IndxDebug25 = Tracers::IndxInvalid;
inline static I4 IndxDebug26 = Tracers::IndxInvalid;
inline static I4 IndxDebug27 = Tracers::IndxInvalid;
inline static I4 IndxDebug28 = Tracers::IndxInvalid;
inline static I4 IndxDebug29 = Tracers::IndxInvalid;
inline static I4 IndxDebug30 = Tracers::IndxInvalid;

// Tracer definitions packaged in a defineAllTracers function
static void defineAllTracers() {

   define("Temperature",                        ///< [in] Name of tracer
          "Conservative Temperature",           ///< [in] Long name, description
          "degree_C",                           ///< [in] Units
          "sea_water_conservative_temperature", ///< [in] CF standard Name
          -273.15,                              ///< [in] min valid field value
          100.0,                                ///< [in] max valid field value
          1.e33,                                ///< [in] value of undef entries
          IndxTemp);                            ///< [out] (optional) static index

   define("Salinity", "Absolute Salinity", "g kg-1",
          "sea_water_absolute_salinity", 0.0, 50.0, 1.e33, IndxSalt);

   define("Debug1", "Debug Tracer 1", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug1);
   define("Debug2", "Debug Tracer 2", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug2);
   define("Debug3", "Debug Tracer 3", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug3);
   define("Debug4", "Debug Tracer 4", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug4);
   define("Debug5", "Debug Tracer 5", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug5);
   define("Debug6", "Debug Tracer 6", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug6);
   define("Debug7", "Debug Tracer 7", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug7);
   define("Debug8", "Debug Tracer 8", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug8);
   define("Debug9", "Debug Tracer 9", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug9);
   define("Debug10", "Debug Tracer 10", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug10);
   define("Debug11", "Debug Tracer 11", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug11);
   define("Debug12", "Debug Tracer 12", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug12);
   define("Debug13", "Debug Tracer 13", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug13);
   define("Debug14", "Debug Tracer 14", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug14);
   define("Debug15", "Debug Tracer 15", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug15);
   define("Debug16", "Debug Tracer 16", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug16);
   define("Debug17", "Debug Tracer 17", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug17);
   define("Debug18", "Debug Tracer 18", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug18);
   define("Debug19", "Debug Tracer 19", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug19);
   define("Debug20", "Debug Tracer 20", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug20);
   define("Debug21", "Debug Tracer 21", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug21);
   define("Debug22", "Debug Tracer 22", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug22);
   define("Debug23", "Debug Tracer 23", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug23);
   define("Debug24", "Debug Tracer 24", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug24);
   define("Debug25", "Debug Tracer 25", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug25);
   define("Debug26", "Debug Tracer 26", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug26);
   define("Debug27", "Debug Tracer 27", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug27);
   define("Debug28", "Debug Tracer 28", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug28);
   define("Debug29", "Debug Tracer 29", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug29);
   define("Debug30", "Debug Tracer 30", "none", "none", 0.0, 100.0, 1.e33,
          IndxDebug30);
}
