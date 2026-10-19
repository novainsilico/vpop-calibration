| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/api/\_\_init\_\_.py                 |        8 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/api/interface.py                    |       56 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/compatibility.py                    |       14 |       12 |        0 |        0 |     14% |      3-19 |
| vpop\_calibration/config.py                           |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |        2 |       12 |        4 |     94% |59, 73, 128-\>131, 167-\>170 |
| vpop\_calibration/metropolis\_hastings.py             |       59 |        1 |        8 |        1 |     97% |       137 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |       23 |       44 |       14 |     80% |51, 53, 55, 57, 171, 236, 284-294, 368, 375, 382, 387, 391, 395, 402-404, 453 |
| vpop\_calibration/model/gp.py                         |      227 |       49 |       68 |       20 |     75% |61, 66, 102, 139, 141-142, 146-149, 224-225, 232, 252, 286, 308, 310, 364, 385, 389-394, 401-409, 411-\>346, 435-437, 465, 475-484, 514-\>exit, 589-596 |
| vpop\_calibration/model/plot.py                       |      122 |       11 |       32 |        8 |     88% |76-77, 147-148, 174-\>177, 219-\>178, 224-225, 244, 253-\>255, 256-259 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      190 |       26 |       46 |        5 |     83% |9-10, 13-14, 147, 152-\>154, 160-164, 252, 273-279, 324-342, 361 |
| vpop\_calibration/pynlme/config.py                    |       14 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/data.py                      |       61 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/diagnostics.py               |      172 |       11 |       28 |        9 |     90% |91-92, 163-\>166, 208, 222-223, 260, 262, 297, 318, 379, 483 |
| vpop\_calibration/pynlme/error\_estimation.py         |       40 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/importance\_sampling.py      |       64 |        0 |       12 |        1 |     99% |   87-\>90 |
| vpop\_calibration/pynlme/indexing.py                  |       66 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/initial\_estimates.py        |       79 |       46 |       22 |        0 |     35% |45-72, 75-78, 84-102, 115-125, 128-133, 136-142 |
| vpop\_calibration/pynlme/model.py                     |      250 |        1 |       28 |        3 |     99% |83-\>90, 242-\>238, 463 |
| vpop\_calibration/pynlme/params.py                    |      180 |        4 |       48 |        4 |     96% |57, 59, 133-\>144, 141, 241 |
| vpop\_calibration/pynlme/plot.py                      |      496 |       43 |      134 |       37 |     87% |158-\>160, 192, 194-195, 241-242, 254, 279, 312-313, 325-\>329, 330, 339-340, 365, 399-400, 415, 419-\>425, 431, 436, 474, 502-503, 516, 558, 560, 573-574, 619, 634-638, 646, 650-651, 658, 780, 789, 791-\>794, 821-\>832, 869-870, 881-885, 969-970, 988, 1114-\>1128, 1131 |
| vpop\_calibration/pynlme/residuals.py                 |      102 |        1 |       10 |        1 |     98% |        51 |
| vpop\_calibration/pynlme/schemas.py                   |       25 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/pynlme/utils.py                     |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       29 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/estimates.py                   |      103 |       21 |       30 |        1 |     74% |161-166, 181-196, 199-206, 226, 230 |
| vpop\_calibration/saem/fixed\_effects.py              |       32 |        1 |        4 |        1 |     94% |        54 |
| vpop\_calibration/saem/m\_step.py                     |       53 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/optimizer.py                   |      165 |       18 |       50 |        6 |     85% |158, 179-182, 185-188, 191-198, 200-\>exit, 436-437, 453-462 |
| vpop\_calibration/saem/plot.py                        |       50 |       39 |       20 |        0 |     16% |3-4, 7-8, 24-59, 62-70, 73-74 |
| vpop\_calibration/saem/scheduler.py                   |       48 |        0 |       14 |        0 |    100% |           |
| vpop\_calibration/saem/utils.py                       |       27 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/config.py                       |        7 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                  |       47 |        4 |       14 |        7 |     82% |47-48, 52, 60, 62-\>65, 67-\>81, 71-\>75, 75-\>81 |
| vpop\_calibration/sdk/model.py                        |       46 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/saem.py                         |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/base.py           |       30 |        1 |        0 |        0 |     97% |        38 |
| vpop\_calibration/structural\_model/gp.py             |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/sbml.py           |       94 |        9 |       14 |        3 |     87% |34-35, 38, 68-73, 76 |
| vpop\_calibration/structural\_model/simwork.py        |      128 |        7 |       22 |        2 |     91% |72-89, 121 |
| vpop\_calibration/utils.py                            |       45 |       18 |       18 |        3 |     51% |29-30, 35, 40-52, 60-61 |
| **TOTAL**                                             | **3496** |  **349** |  **722** |  **131** | **86%** |           |
