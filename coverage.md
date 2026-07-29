| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/\_\_init\_\_.py                     |        6 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/config.py                           |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |       64 |       12 |        0 |     26% |30-31, 38-64, 73-83, 92-98, 111-119, 129-158, 168-217 |
| vpop\_calibration/interface.py                        |       56 |       26 |        2 |        0 |     52% |37-52, 55-60, 69-86, 89-91, 100-102 |
| vpop\_calibration/metropolis\_hastings.py             |       54 |       37 |        8 |        0 |     27% |18-30, 34-51, 54-66, 89-147 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |      138 |       44 |        0 |     12% |46-209, 233-255, 261-266, 271-276, 283-293, 298-304, 311-337, 361-410, 430-453 |
| vpop\_calibration/model/gp.py                         |      227 |      195 |       68 |        0 |     11% |55-142, 145-153, 216-291, 297-417, 431-438, 444-453, 457-468, 475-484, 488-492, 497-526, 548-553, 563-564, 571-576, 585-586, 591-598 |
| vpop\_calibration/model/plot.py                       |      124 |      111 |       32 |        0 |      8% |25-79, 86-162, 172-239, 244-257, 265-272 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      181 |      135 |       44 |        0 |     20% |25, 29, 37-46, 54-58, 63-85, 88-97, 103-118, 121-141, 144-161, 165-191, 194-196, 199-225, 229-252, 256-260, 264-265, 269-279, 283-293, 297-305, 312-317, 321-324 |
| vpop\_calibration/pynlme/config.py                    |       13 |        2 |        0 |        0 |     85% |    14, 18 |
| vpop\_calibration/pynlme/data.py                      |       46 |       36 |        8 |        0 |     19% |18-65, 83-106, 118-127 |
| vpop\_calibration/pynlme/diagnostics.py               |      164 |      134 |       22 |        0 |     16% |34-45, 48-53, 59-66, 72, 83-145, 156-220, 223-278, 281-292, 296-307, 316-383, 388-395 |
| vpop\_calibration/pynlme/importance\_sampling.py      |       56 |       41 |       10 |        0 |     23% |12-14, 17-28, 34-48, 53-63, 67-71, 74-79, 83-99 |
| vpop\_calibration/pynlme/indexing.py                  |       60 |       34 |        8 |        0 |     38% |19-22, 29-44, 60-76, 87-97, 110-139 |
| vpop\_calibration/pynlme/initial\_estimates.py        |       78 |       61 |       22 |        0 |     17% |15-37, 43, 46-71, 74-77, 83-101, 114-124, 127-132, 135-141 |
| vpop\_calibration/pynlme/model.py                     |      260 |      209 |       34 |        0 |     17% |28, 35-41, 44-53, 75-191, 196-205, 214-232, 239-249, 256-278, 285-295, 303-311, 316-324, 329-337, 340-347, 358-362, 371-372, 387-394, 402-406, 418-427, 442-451, 458-469, 485-495, 500-526, 540-543, 548-551, 565-581, 586-607, 616-661, 664-668 |
| vpop\_calibration/pynlme/params.py                    |      112 |       39 |       16 |        2 |     59% |23, 27-28, 35-39, 53-57, 62, 84, 114, 119, 124, 129, 132-145, 152-157, 162, 166-167 |
| vpop\_calibration/pynlme/plot.py                      |      420 |      395 |      132 |        0 |      5% |24, 32-125, 132-191, 202-275, 287-374, 384-489, 498-528, 545-646, 652-735, 745-835, 844-904 |
| vpop\_calibration/pynlme/residuals.py                 |       61 |       51 |       12 |        0 |     14% |24-53, 64-79, 90-112, 126-157, 167-177 |
| vpop\_calibration/pynlme/schemas.py                   |       12 |        1 |        0 |        0 |     92% |        14 |
| vpop\_calibration/pynlme/utils.py                     |       26 |       16 |        2 |        0 |     36% |22-53, 61-78, 82-85 |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       27 |        2 |        0 |        0 |     93% |    43, 47 |
| vpop\_calibration/saem/estimates.py                   |       84 |       51 |       20 |        0 |     32% |17, 21, 29-42, 49-60, 74-81, 94-111, 122-131, 134-141, 144-152, 156, 160 |
| vpop\_calibration/saem/m\_step.py                     |       53 |       35 |        2 |        0 |     33% |23-30, 41-55, 58, 69-84, 87-96, 99-104, 117-135, 141-164 |
| vpop\_calibration/saem/optimizer.py                   |      134 |      104 |       42 |        0 |     17% |32-61, 66-91, 100-118, 124-143, 146-174, 177-184, 193-305, 310-337, 344-360, 363-371 |
| vpop\_calibration/saem/plot.py                        |       41 |       32 |       14 |        0 |     16% |17-50, 53-61, 64 |
| vpop\_calibration/saem/scheduler.py                   |       50 |       34 |       18 |        0 |     24% |16-24, 27-29, 33, 37-42, 46-55, 59-70, 73, 85-94 |
| vpop\_calibration/saem/utils.py                       |       27 |       21 |        0 |        0 |     22% |16-21, 38-39, 44-47, 53-65, 73-84 |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        4 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                  |       43 |       20 |       12 |        0 |     42% |     35-78 |
| vpop\_calibration/sdk/model.py                        |       20 |       13 |        0 |        0 |     35% |24-45, 49-51, 66-86 |
| vpop\_calibration/sdk/saem.py                         |        5 |        2 |        0 |        0 |     60% |      7-10 |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        6 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |       30 |        2 |        0 |     26% |22-89, 101-143 |
| vpop\_calibration/structural\_model/base.py           |       30 |       21 |        0 |        0 |     30% |27-31, 38, 43-75 |
| vpop\_calibration/structural\_model/gp.py             |       26 |       19 |        2 |        0 |     25% |17-30, 39-58 |
| vpop\_calibration/structural\_model/sbml.py           |       81 |       64 |        8 |        0 |     19% |21-31, 42-105, 117-158, 161-176, 183-214 |
| vpop\_calibration/structural\_model/simwork.py        |      128 |       94 |       22 |        0 |     23% |40-55, 67-91, 99-130, 135-156, 161-190, 200-256, 268-309, 319-343, 350-386 |
| vpop\_calibration/utils.py                            |       24 |       15 |        6 |        0 |     30% |13-16, 22-30, 34-36 |
| **TOTAL**                                             | **3044** | **2283** |  **626** |    **3** | **21%** |           |
