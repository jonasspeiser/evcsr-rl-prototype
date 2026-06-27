# Datasets

## ASFINAG data

**Source**: Autobahnen- und Schnellstraßen-Finanzierungs-Aktiengesellschaft, Österreich  
**Link**: https://www.asfinag.at/verkehr-sicherheit/verkehrszaehlung/

**Description**:  
Public data from traffic counts in Austria for the year 2023. Granularity: number of vehicles per month.

**Usage in project**:  
This data was only used for comparison purposes as its resolution is not high enough.

## BASt data

**Source**: Bundesanstalt für Straßenwesen, Deutschland  
**Link**: https://www.bast.de/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Stundenwerte.html  
**Link neu**: https://www.bast.de/DE/Themen/Digitales/HF_1/Massnahmen/verkehrszaehlung/Stundenwerte.html?nn=414410

**Description**:  
Public data from highway traffic counts in Germany for the year 2022. Granularity: number of vehicles per hour.

**Usage in project**:  
Realistic distributions of traffic density over time for simulation scenarios.

## OBELIS data

**Source**: NOW GmbH, Nationale Leitstelle Ladeinfrastruktur, Deutschland, https://nationale-leitstelle.de/verstehen/   
**Link**: https://mobilithek.info/offers/714073450865197056

**Description**:  

Charging processes that have been submitted via OBELIS as part of the semi-annual reports for subsidized charging points. Currently, this includes charging processes up to and including the first half of 2024. The reporting of charging processes for the second half of 2024 is still ongoing. These will be included in the next update.

To prevent the association of charging process data with charging point and charging station data—due to the legitimate confidentiality interests of the operators—the charging point IDs and charging station IDs in the charging process table are overwritten using a dictionary filled with random numbers. To make clear that these are not the original IDs but randomly shuffled ones, the suffix “_shuffled” is added. This ensures that charging processes that occurred at the same charging point are still assigned to the same (shuffled) charging point. However, since the IDs are overwritten with random numbers, the charging processes are assigned to random charging points and not the actual ones at which they took place. Therefore, no conclusions can be drawn about trade secrets.

To allow at least limited spatial analysis of utilization, information about the federal state and the location of the actual charging station is attached to the charging processes before the “shuffling” of the charging station IDs. More precise information such as postal code or city is not provided due to confidentiality concerns.


**Usage in project**:  
Used to simulate realistic charging station occupancy patterns (for non-observable vehicles).