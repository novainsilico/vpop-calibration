| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/\_\_init\_\_.py                     |        6 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/config.py                           |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |       64 |       12 |        0 |     26% |30-31, 38-64, 73-83, 92-98, 111-119, 129-158, 168-217 |
| vpop\_calibration/interface.py                        |       50 |       23 |        2 |        0 |     52% |34-47, 50-55, 64-78, 81-83, 92-94 |
| vpop\_calibration/metropolis\_hastings.py             |       54 |       37 |        8 |        0 |     27% |18-30, 34-51, 54-66, 89-147 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |      138 |       44 |        0 |     12% |46-209, 233-255, 261-266, 271-276, 283-293, 298-304, 311-337, 361-410, 430-453 |
| vpop\_calibration/model/gp.py                         |      227 |      195 |       68 |        0 |     11% |55-142, 145-153, 216-291, 297-417, 431-438, 444-453, 457-468, 475-484, 488-492, 497-526, 548-553, 563-564, 571-576, 585-586, 591-598 |
| vpop\_calibration/model/plot.py                       |      124 |      111 |       32 |        0 |      8% |25-79, 86-162, 172-239, 244-257, 265-272 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      178 |      133 |       42 |        0 |     20% |24, 28, 36-45, 53-57, 62-84, 87-96, 102-117, 120-138, 141-158, 162-188, 191-193, 196-222, 226-249, 253-257, 261-262, 266-276, 280-290, 294-302, 309-312, 316-319 |
| vpop\_calibration/pynlme/config.py                    |       12 |        2 |        0 |        0 |     83% |    12, 16 |
| vpop\_calibration/pynlme/data.py                      |       46 |       36 |        8 |        0 |     19% |18-65, 83-106, 118-127 |
| vpop\_calibration/pynlme/diagnostics.py               |      156 |      128 |       20 |        0 |     16% |33-40, 43-45, 51-55, 61, 72-134, 145-209, 212-267, 270-281, 285-296, 305-372 |
| vpop\_calibration/pynlme/indexing.py                  |       60 |       34 |        8 |        0 |     38% |19-22, 29-44, 60-76, 87-97, 110-139 |
| vpop\_calibration/pynlme/model.py                     |      249 |      199 |       28 |        0 |     18% |28, 35-41, 44-53, 75-189, 194-199, 208-220, 227-237, 244-266, 273-283, 291-299, 304-312, 317-325, 336-340, 349-350, 365-372, 380-384, 396-405, 420-429, 436-447, 463-473, 478-504, 518-521, 526-529, 543-559, 564-585, 594-639, 642-646 |
| vpop\_calibration/pynlme/params.py                    |      112 |       39 |       16 |        2 |     59% |23, 27-28, 35-39, 53-57, 62, 84, 114, 119, 124, 129, 132-145, 152-157, 162, 166-167 |
| vpop\_calibration/pynlme/plot.py                      |      410 |      387 |      134 |        0 |      4% |22, 30-123, 130-189, 200-273, 285-372, 382-487, 496-526, 543-644, 650-733, 743-815, 824-881 |
| vpop\_calibration/pynlme/residuals.py                 |       61 |       51 |       12 |        0 |     14% |24-53, 64-79, 90-112, 126-157, 167-177 |
| vpop\_calibration/pynlme/schemas.py                   |       13 |        1 |        0 |        0 |     92% |        15 |
| vpop\_calibration/pynlme/utils.py                     |       18 |       12 |        0 |        0 |     33% |17-48, 56-73 |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       27 |        2 |        0 |        0 |     93% |    43, 47 |
| vpop\_calibration/saem/estimates.py                   |       84 |       51 |       20 |        0 |     32% |17, 21, 29-42, 49-60, 74-81, 94-111, 122-131, 134-141, 144-152, 156, 160 |
| vpop\_calibration/saem/m\_step.py                     |       53 |       35 |        2 |        0 |     33% |23-30, 41-55, 58, 69-84, 87-96, 99-104, 117-135, 141-164 |
| vpop\_calibration/saem/optimizer.py                   |      146 |      115 |       46 |        0 |     16% |32-61, 66-91, 100-118, 124-143, 146-174, 177-184, 193-299, 304-331, 335-346, 353-368, 371-379 |
| vpop\_calibration/saem/plot.py                        |       41 |       32 |       14 |        0 |     16% |17-50, 53-61, 64 |
| vpop\_calibration/saem/scheduler.py                   |       50 |       34 |       18 |        0 |     24% |16-24, 27-29, 33, 37-42, 46-55, 59-70, 73, 85-94 |
| vpop\_calibration/saem/utils.py                       |       27 |       21 |        0 |        0 |     22% |16-21, 38-39, 44-47, 53-65, 73-84 |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        4 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                  |       33 |       12 |        8 |        0 |     51% |     33-58 |
| vpop\_calibration/sdk/model.py                        |       20 |       13 |        0 |        0 |     35% |24-45, 49-51, 66-86 |
| vpop\_calibration/sdk/saem.py                         |        5 |        2 |        0 |        0 |     60% |      7-10 |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |       30 |        2 |        0 |     26% |22-89, 101-143 |
| vpop\_calibration/structural\_model/base.py           |       29 |       20 |        0 |        0 |     31% |27-30, 37, 42-74 |
| vpop\_calibration/structural\_model/gp.py             |       26 |       19 |        2 |        0 |     25% |17-30, 39-58 |
| vpop\_calibration/structural\_model/simwork.py        |      127 |       93 |       20 |        0 |     23% |40-55, 66-88, 96-127, 132-153, 158-187, 197-253, 265-306, 316-340, 347-383 |
| vpop\_calibration/utils.py                            |        6 |        3 |        2 |        0 |     38% |      9-12 |
| **TOTAL**                                             | **2764** | **2073** |  **570** |    **3** | **21%** |           |
