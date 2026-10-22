| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/api/\_\_init\_\_.py                 |        9 |        0 |        0 |        0 |    100% |           |
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
| vpop\_calibration/pynlme/conditional\_distribution.py |      190 |       33 |       46 |        5 |     79% |9-10, 13-14, 147, 152-\>154, 160-164, 252, 273-279, 310-320, 324-342, 361 |
| vpop\_calibration/pynlme/config.py                    |       14 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/data.py                      |       61 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/diagnostics.py               |      178 |       16 |       28 |        9 |     88% |91-92, 163-\>166, 208, 222-223, 260, 262, 297, 318, 379, 392-400, 494 |
| vpop\_calibration/pynlme/error\_estimation.py         |       40 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/importance\_sampling.py      |       64 |        0 |       12 |        1 |     99% |   87-\>90 |
| vpop\_calibration/pynlme/indexing.py                  |       66 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/initial\_estimates.py        |       79 |       61 |       22 |        0 |     18% |15-37, 45-72, 75-78, 84-102, 115-125, 128-133, 136-142 |
| vpop\_calibration/pynlme/model.py                     |      250 |        1 |       28 |        3 |     99% |83-\>90, 242-\>238, 463 |
| vpop\_calibration/pynlme/params.py                    |      180 |        4 |       48 |        4 |     96% |57, 59, 133-\>144, 141, 241 |
| vpop\_calibration/pynlme/plot.py                      |      431 |       45 |      126 |       37 |     85% |37-\>39, 71, 73-74, 120-121, 133, 158, 191-192, 204-\>208, 209, 218-219, 244, 278-279, 294, 298-\>304, 310, 315, 353, 381-382, 395, 437, 439, 452-453, 498, 513-517, 525, 529-530, 537, 659, 668, 670-\>673, 700-\>711, 748-749, 760-764, 848-849, 874, 961-962, 1006 |
| vpop\_calibration/pynlme/residuals.py                 |      102 |        1 |       10 |        1 |     98% |        51 |
| vpop\_calibration/pynlme/schemas.py                   |       25 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/pynlme/utils.py                     |       26 |        4 |        2 |        0 |     79% |     82-85 |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       34 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/estimates.py                   |      103 |       21 |       30 |        1 |     74% |161-166, 181-196, 199-206, 226, 230 |
| vpop\_calibration/saem/fixed\_effects.py              |       32 |        1 |        4 |        1 |     94% |        54 |
| vpop\_calibration/saem/m\_step.py                     |       53 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/optimizer.py                   |      170 |       22 |       56 |        7 |     82% |158-159, 180-183, 186-189, 192-194, 197-204, 206-\>exit, 442-443, 459-468 |
| vpop\_calibration/saem/plot.py                        |       50 |       39 |       20 |        0 |     16% |3-4, 7-8, 24-59, 62-70, 73-74 |
| vpop\_calibration/saem/scheduler.py                   |       48 |        0 |       14 |        0 |    100% |           |
| vpop\_calibration/saem/utils.py                       |       27 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/base.py           |       30 |        1 |        0 |        0 |     97% |        38 |
| vpop\_calibration/structural\_model/gp.py             |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/sbml.py           |       94 |        9 |       14 |        3 |     87% |34-35, 38, 68-73, 76 |
| vpop\_calibration/structural\_model/simwork.py        |      128 |        7 |       22 |        2 |     91% |72-89, 121 |
| vpop\_calibration/utils.py                            |       47 |       18 |       18 |        3 |     52% |42-43, 48, 66-78, 86-87 |
| **TOTAL**                                             | **3340** |  **382** |  **706** |  **125** | **85%** |           |
