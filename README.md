# ISSCC 2024 CIM 论文合集（Paper with links)

**ISSCC 2024 decisions are now available on OpenReview！**

注1：欢迎各位大佬提交issue，分享CIM 论文和开源项目！

有任何问题，欢迎在Discussion 发表讨论，第一时间分享最新最前沿的存内计算、存储介质，存内应用，深度学习等学起来！



## 目录

##### [存内计算ISSCC 34系列](存内计算 ISSCC 34系列:)

##### [存内计算基础概念](存内计算- 基础概念:)

#####  [存内计算忆阻器论文合集](存内计算- 忆阻器论文合集:)

#####  [存内计算-向量乘矩阵](存内计算- 向量乘矩阵:)

#####  [存内计算-数字与模拟计算](存内计算- 数字与模拟计算:)

#####  [存内计算-现有应用场景](存内计算- 现有应用场景:)

#####  [存内计算-存内计算研究进展](存内计算- 存内计算研究进展:)

#####  [存内计算-存内计算在机器学习领域作用](存内计算- 存内计算在机器学习领域作用:)

#####  [存内计算-存内计算加速器](存内计算- 存内计算加速器:)



## 存内计算 ISSCC 34系列：

ISSCC 34系列论文涵盖了多个前沿研究领域，其中包括当前最前沿的研究领域，存内计算技术，数字电路设计，模拟电路设计，混合信号设计，CMOS图像传感器，电容数字转换器等多个领域,此系列包含ISSCC 34系列的每节论文以及解析，共开发者学习以及讨论。



###### ISSCC.34.1 ——适用于高精度 AI 应用的 28nm 83.23TFLOPS/W POSIT 

Paper: [10.1109/ISSCC49657.2024.10454567](http://dx.doi.org/10.1109/ISSCC49657.2024.10454567)

###### 解析： https://blog.csdn.net/m0_58966968/article/details/139472261?utm_source=bbs_include



###### ISSCC 34.2——双端口设计实现高面积利用的浮点/整数存算

Paper: [10.1109/ISSCC49657.2024.10454447](https://doi.org/10.1109/ISSCC49657.2024.10454447)

解析：https://blog.csdn.net/m0_58966968/article/details/139983166?spm=1001.2014.3001.5502



###### ISSCC 34.3——“闪电”数模混合存内计算，适应transformer和CNNs架构

Paper: [10.1109/ISSCC19947.2020.9062931](https://doi.org/10.1109/ISSCC19947.2020.9062931)

解析：https://blog.csdn.net/m0_58966968/article/details/138122847?spm=1001.2014.3001.5502



###### ISSCC 34.4—— 采用台积电3nm工艺制作的数字存算一体芯片

Paper: [10.1109/ISSCC49657.2024.10454556](https://doi.org/10.1109/ISSCC49657.2024.10454556)

解析：https://bbs.csdn.net/topics/618245914



###### ISSCC 34.5——用于统一加速CNN和Transformers的818-4094 TOPS/W电容可重构CIM宏单元

Paper: [10.1109/ISSCC49657.2024.10454489](https://doi.org/10.1109/ISSCC49657.2024.10454489)

解析：https://bbs.csdn.net/topics/619197013



## 存内计算-基础概念：

存内计算（Computing in Memory）是指将计算单元直接嵌入到存储器中，顾名思义就是把计算单元嵌入到内存当中，通常计算机运行的冯·诺依曼体系包括存储单元和计算单元两部分。在本质上消除不必要的数据搬移的延迟和功耗，从而消除了传统的冯·诺依曼架构的瓶颈，打破存储墙。据悉，存内计算特别适用于需要大数据处理的领域，比如云计算、人工智能等领域，最重要的一点是存内计算是基于存储介质的计算架构，而且存内计算是一种新型存储架构且轻松打破传统存储架构的瓶颈。



###### **1.In-Memory Computing: A Game Changer for Enterprise Applications**

Paper:  https://www.usenix.org/conference/atc19/presentation/lee



###### 2.In-Memory Computing with Phase-Change Memory: Opportunities and Challenges

Links: https://www.sciencedirect.com/science/article/abs/pii/S0165562520301085



###### 3.Towards End-to-End In-Memory Computing Systems

Links: https://link.springer.com/article/10.1007/s13389-020-00308-5



## 存内计算-忆阻器论文合集：

存内计算忆阻器是一种前沿技术，它利用忆阻器的电阻可变特性，在存储单元中直接完成计算操作，避免了数据在内存和处理器之间的频繁传输，提高了计算效率和能效比。忆阻器作为存内计算的重要器件候选，可以支持各种模拟计算应用，包括人工神经网络（ANN）、机器学习、科学计算和数字图像处理等。下面论文包括几种存内计算领域热门关注的忆阻器SRAM，MRAM，Nor Flash, DRAM 。



#### SRAM: 

###### 1.In-Memory Computing with Emerging Non-Volatile Memory: A Machine Learning Perspective

Paper: https://www.nature.com/articles/s41729-021-00100-5



###### 2. A 137.5 TOPS/W SRAM Compute-in-Memory Macro with 9-b Memory Cell-Embedded ADCs and Signal Margin Enhancement Techniques for AI Edge Applications

Paper:https://arxiv.org/abs/2307.05944



###### 3.A Charge Domain P-8T SRAM Compute-In-Memory with Low-Cost DAC/ADC Operation for 4-bit Input Processing

Paper: https://arxiv.org/abs/2211.16008



#### MRAM：

###### 1.MRAM-based Analog Sigmoid Function for In-memory Computing

Paper ：https://arxiv.org/abs/2204.09918



###### 2.A noise-tolerant, resource-saving probabilistic binary neural network implemented by the SOT-MRAM compute-in-memory system

Paper: https://arxiv.org/abs/2403.19374



#### Nor Flash: 

###### 1.Collaborative Training for Compensation of Inference Errors in NOR Flash Computing in memory Chips

Paper: [10.1109/CSCWD61410.2024.10580459](https://doi.org/10.1109/CSCWD61410.2024.10580459)



###### 2.A 40nm 1Mb 35.6 TOPS/W MLC NOR-Flash Based Computation-in-Memory Structure for Machine Learning

Paper:[10.1109/ISCAS51556.2021.9401600](https://doi.org/10.1109/ISCAS51556.2021.9401600)



###### 3.Multibit Content Addressable Memory Design and Optimization Based on 3-D nand-Compatible IGZO Flash

Paper: https://xplorestaging.ieee.org/document/10521670



###### 4.Design-Technology Co-Optimizations (DTCO) for General-Purpose Computing In-Memory Based on 55nm NOR Flash Technology

Paper:  [10.1109/IEDM19574.2021.9720625](https://doi.org/10.1109/IEDM19574.2021.9720625)



#### DRAM：

###### 1.Computing-In-Memory Using 1T1C Embedded DRAM Cell with Micro Sense Amplifier for Enhancing Throughput

Paper:[10.1109/ICCE-Asia57006.2022.9954870](https://doi.org/10.1109/ICCE-Asia57006.2022.9954870)

## 

###### 2.IGZO CIM: Enabling In-Memory Computations Using Multilevel Capacitorless Indium–Gallium–Zinc–Oxide-Based Embedded DRAM Technology

Paper:  [10.1109/JXCDC.2022.3188366](https://doi.org/10.1109/JXCDC.2022.3188366)



###### 3.A Novel Transpose 2T-DRAM based Computing-in-Memory Architecture for On-chip DNN Training and Inference

Paper: [10.1109/AICAS57966.2023.10168641](https://doi.org/10.1109/AICAS57966.2023.10168641)



## 存内计算-向量乘矩阵：

所有存内计算操作中, 最普遍的是利用基尔霍夫定律 (Kirchoff’s Law) 进行向量乘矩阵操作. 原因在于: （1）它能够高效地将计算和存储紧密结合; (2) 它的计算效率高 (即, 在一个读操作延迟内能 完成一次向量乘矩阵); (3) 目前流行的数据密集型应用中, 如机器学习应用和图计算应用, 向量乘矩阵的计算占了总计算量的 90% 以上 。



###### 1.Dot-Product Engine for Neuromorphic Computing: Programming 1T1M Crossbar to Accelerate Matrix-Vector Multiplication

Paper: https://dl.acm.org/doi/abs/10.1145/2897937.2898010



###### 2.Challenges and Trends of Nonvolatile In-Memory-Computation Circuits for AI Edge Devices

Paper: https://ieeexplore.ieee.org/abstract/document/9586071



###### 3.Challenges and Trends of SRAM-Based Computing-In-Memory for AI Edge Devices

Paper: https://ieeexplore.ieee.org/abstract/document/9382915



###### 4.Scalable and Programmable Neural Network Inference Accelerator Based on In-Memory Computing

Paper: https://ieeexplore.ieee.org/abstract/document/9580448



###### 5.A Programmable Heterogeneous Microprocessor Based on Bit-Scalable In-Memory Computing

Paper: https://ieeexplore.ieee.org/abstract/document/9082159



###### 6.Victor: A Variation-resilient Approach Using CellClustered Charge-domain computing for Highdensity High-throughput MLC CiM

Paper: https://ieeexplore.ieee.org/abstract/document/10247934



###### 7.DIMCA: An Area-Efficient Digital In-Memory Computing Macro Featuring Approximate Arithmetic Hardware in 28 nm

Paper: https://ieeexplore.ieee.org/abstract/document/10266328



###### 8.ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars

Paper: https://dl.acm.org/doi/abs/10.1145/3007787.3001139



###### 9.Specific ADC of NVM-Based Computation-in-Memory for Deep Neural Networks

Paper: https://ieeexplore.ieee.org/abstract/document/10668403



###### 10.A 7-nm Compute-in-Memory SRAM Macro Supporting Multi-Bit Input, Weight and Output and Achieving 351 TOPS/W and 372.4 GOPS

Paper: https://ieeexplore.ieee.org/abstract/document/9250531



## 存内计算-数字与模拟计算：

数字存内计算与模拟存内计算各有优劣，都是存算一体发展进程中的重点发展路径，数字存内计算由于其高速、高精度、抗噪性强、工艺技术成熟、能效比高等特点，更适用于大算力、云计算、边缘计算等应用场景；模拟存内计算由于其非易失性、高密度、低成本、功耗低等特点，更适用于小算力、端侧、需长时待机等的应用场景。在如今可穿戴设备、智能家具、玩具机器人等应用走进千家万户的背景下，模拟存内计算的高能效、小面积、低成本等市场优势逐渐凸显，比如前面所提到的知存科技WTM2101已率先进入市场规模化应用，在商业化进程中处于领先地位，且更高算力WTM-8系列即将量产，在端侧AI市场具有极大的应用潜力。



#### Digital CIM: 

###### 1.An 89TOPS/W and 16.3TOPS/mm2 All-Digital SRAM-Based Full-Precision Compute-In Memory Macro in 22nm for Machine-Learning Edge Applications

Paper: https://ieeexplore.ieee.org/abstract/document/9365766



###### 2.A 5-nm 254-TOPS/W 221-TOPS/mm2 Fully-Digital Computingin-Memory Macro Supporting Wide-Range Dynamic-VoltageFrequency Scaling and Simultaneous MAC and Write Operations

Paper: https://ieeexplore.ieee.org/abstract/document/9731754



###### 3.A 28nm 64-kb 31.6-TFLOPS/W Digital-Domain Floating-PointComputing-Unit and Double-Bit 6T-SRAM Computing-inMemory Macro for Floating-Point CNNs

Paper: https://ieeexplore.ieee.org/abstract/document/10067260



#### Analog CIM: 

###### 1.End-to-End DNN Inference on a Massively Parallel Analog In Memory Computing Architecture

Paper: https://ieeexplore.ieee.org/abstract/document/10137208



###### 2.A 22nm Delta-Sigma Computing-In-Memory (ΔΣCIM) SRAM Macro with Near-Zero-Mean Outputs and LSB-First ADCs Achieving 21.38TOPS/W for 8b-MAC Edge AI Processing

Paper: https://ieeexplore.ieee.org/abstract/document/10067289



###### 3.CiM-BNN:Computing-in-MRAM Architecture for Stochastic Computing based Bayesian Neural Network

Paper: https://ieeexplore.ieee.org/abstract/document/10262235



#### Hybrid Analog-Digital CIM: 

###### 1.A 22nm 64kb Lightning-Like Hybrid Computing-in-Memory Macro with a Compressed Adder Tree and Analog-Storage Quantizers for Transformer and CNNs

Paper: https://ieeexplore.ieee.org/abstract/document/10454278



###### 2.A 28nm 72.12TFLOPS/W Hybrid-Domain Outer-Product Based Floating-Point SRAM Computing-in-Memory Macro with Logarithm Bit-Width Residual ADC

Paper: https://ieeexplore.ieee.org/abstract/document/10454313



###### 3.Hybrid Analog-Digital In-Memory Computing

Paper: https://ieeexplore.ieee.org/abstract/document/9643526



###### 4.HD-CIM: Hybrid-Device Computing-In-Memory Structure Based on MRAM and SRAM to Reduce Weight Loading Energy of Neural Networks

Paper: https://ieeexplore.ieee.org/abstract/document/9865238



## 存内计算- 现有应用场景：

存内计算技术正迅速成为人工智能和其他数据密集型应用的关键驱动力。它通过将计算任务直接嵌入到存储单元中，显著减少了数据在处理器和存储器之间传输的需要，从而降低了延迟和能耗。存内计算在深度学习，图计算，智能传感器以及物联网设备中都有较多应用。



###### 1.A Survey on In-Memory Computing Architectures for Big Data Applications

Paper: https://ieeexplore.ieee.org/document/9150090



###### 2.A Comparative Study of In-Memory Computing Technologies for Data-Intensive Applications

Paper: https://www.sciencedirect.com/science/article/pii/S0167739X21000286



###### 3.In-Memory Computing with Phase-Change Memory: Opportunities and Challenges

Paper: https://www.sciencedirect.com/science/article/abs/pii/S0165562520301085



###### 4.Compute-in-Memory Chips for Deep Learning: Recent Trends and Prospects

Paper: https://ieeexplore.ieee.org/abstract/document/9512855



###### 5.Deep learning acceleration based on in-memory computing

Paper: https://ieeexplore.ieee.org/abstract/document/8865099



###### 6.A Review of In-Memory Computing Architectures for Machine Learning Applications

Paper: https://dl.acm.org/doi/abs/10.1145/3386263.3407649



###### 7.YinMem: A distributed parallel indexed in-memory computation system for large scale data analytics

Paper: https://ieeexplore.ieee.org/abstract/document/7840607



###### 8.Real-Time Awareness Scheduling for Multimedia Big Data Oriented In-Memory Computing

Paper: https://ieeexplore.ieee.org/abstract/document/8283555



###### 9.ReRAM-Based In-Memory Computing for Search Engine and Neural Network Applications

Paper: https://ieeexplore.ieee.org/abstract/document/8681546



###### 10.A 351TOPS/W and 372.4GOPS Compute-in-Memory SRAM Macro in 7nm FinFET CMOS for Machine-Learning Applications

Paper: https://ieeexplore.ieee.org/abstract/document/9062985



## 存内计算- 存内计算研究进展:

存内计算技术因其存储与计算一体化的特有优势，吸引了学术界和工业界的广泛关注。近年来，存内计算的研究进展主要集中在模拟与数字电路的融合、新型存储器件的开发以及提高计算精度和速度等方面。特别是在AI任务中，存内计算展现了卓越的能效与性能，推动了大规模神经网络加速器的设计和应用，以下是存内计算研究进展相关的高质量论文。



###### 1.A full spectrum of computing-in-memory technologies

Paper: https://www.nature.com/articles/s41928-023-01053-4 (https://doi.org/10.1038/s41928-023-01053-4)



###### 2.Trends and challenges in the circuit and macro of RRAM-based computing-in-memory systems

Paper: https://www.sciencedirect.com/science/article/pii/S2709472322000028 (https://doi.org/10.1016/j.chip.2022.100004)



###### 3.A review on SRAM-based computing in-memory: Circuits, functions, and applications

Paper: https://iopscience.iop.org/article/10.1088/1674-4926/43/3/031401/meta (https://doi.org/10.1088/1674-4926/43/3/031401)



###### 4.Memory devices and applications for in-memory computing

Paper: https://www.nature.com/articles/s41565-020-0655-z (https://doi.org/10.1038/s41565-020-0655-z)



###### 5.In-Memory Computing: Advances and Prospects

Paper: https://ieeexplore.ieee.org/abstract/document/8811809 (https://doi.org/10.1109/MSSC.2019.2922889)



###### 6.Analog content-addressable memories with memristors

Paper: https://www.nature.com/articles/s41467-020-15254-4 (https://doi.org/10.1038/s41467-020-15254-4)



###### 7.An area-efficient in-memory implementation method of arbitrary Boolean function based on SRAM array

Paper: https://ieeexplore.ieee.org/abstract/document/10202194 (https://doi.org/10.1109/TC.2023.3301156)



###### 8.In-memory computing with resistive switching devices

Paper: https://www.nature.com/articles/s41928-018-0092-2 (https://doi.org/10.1038/s41928-018-0092-2)



###### 9.Mixed-precision in-memory computing

Paper: https://www.nature.com/articles/s41928-018-0054-8 (https://doi.org/10.1038/s41928-018-0054-8)



###### 10.A survey of SRAM-based in-memory computing techniques and applications

Paper: https://www.sciencedirect.com/science/article/pii/S1383762121001909 (https://doi.org/10.1016/j.sysarc.2021.102276)



## 存内计算- 存内计算在机器学习领域作用:

存内计算技术通过将计算直接在存储单元中进行，大幅减少数据传输延迟和能耗，显著提升了机器学习模型的推理和训练效率。它特别适用于加速神经网络的矩阵运算，增强实时处理能力和能效​，以下是有关存内计算在机器学习领域作用相关的高质量论文。



###### 1.FloatPIM: in-memory acceleration of deep neural network training with high precision

Paper: https://dl.acm.org/doi/abs/10.1145/3307650.3322237 (https://doi.org/10.1145/3307650.3322237)



###### 2.In-Memory Computing in Emerging Memory Technologies for Machine Learning: An Overview

Paper: https://ieeexplore.ieee.org/abstract/document/9218505 (https://doi.org/10.1109/DAC18072.2020.9218505)



###### 3.A 351TOPS/W and 372.4GOPS Compute-in-Memory SRAM Macro in 7nm FinFET CMOS for Machine-Learning Applications

Paper: https://ieeexplore.ieee.org/document/9062985 (https://doi.org/10.1109/ISSCC19947.2020.9062985)



###### 4.A 1.041-Mb/mm2 27.38-TOPS/W Signed-INT8 Dynamic-LogicBased ADC-less SRAM Compute-In-Memory Macro in 28nm with Reconfigurable Bitwise Operation for AI and Embedded Applications

Paper: https://ieeexplore.ieee.org/abstract/document/9731545 (https://doi.org/10.1109/ISSCC42614.2022.9731545)



###### 5.Edge learning using a fully integrated neuro-inspired memristor chip

Paper: https://www.science.org/doi/abs/10.1126/science.ade3483 (https://doi.org/10.1126/science.ade3483)



###### 6.An Energy-Efficient Computing-in-Memory Neuromorphic System with On-Chip Training

Paper: https://ieeexplore.ieee.org/abstract/document/8918995 (https://doi.org/10.1109/BIOCAS.2019.8918995)



###### 7.A High-Density and Reconfigurable SRAM-Based Digital Compute-In-Memory Macro for Low-Power AI Chips

Paper: https://ieeexplore.ieee.org/document/10124240 (https://doi.org/10.1109/TCSII.2023.3276169)



###### 8.Challenges and Trends of SRAM-Based Computing-In-Memory for AI Edge Devices

Paper: https://ieeexplore.ieee.org/document/10124240 (https://doi.org/10.1109/TCSI.2021.3064189)



###### 9.A High Energy Efficient Reconfigurable Hybrid Neural Network Processor for Deep Learning Applications

Paper: https://ieeexplore.ieee.org/abstract/document/8207783 (https://doi.org/10.1109/JSSC.2017.2778281)



###### 10.A Twin-8T SRAM Computation-In-Memory Macro for Multiple-Bit CNN-Based Machine Learning

Paper: https://ieeexplore.ieee.org/abstract/document/8662392 (https://doi.org/10.1109/ISSCC.2019.8662392)



## 存内计算- 存内计算加速器:

存内计算加速器通过将计算与存储功能相结合，减少了数据传输的瓶颈，显著提升了机器学习任务中的能效和处理速度，尤其适用于加速神经网络中的矩阵运算，以下是存内计算加速器相关的高质量论文。



###### 1.A 1–8b Reconfigurable Digital SRAM Compute-in-Memory Macro for Processing Neural Networks

Paper: https://ieeexplore.ieee.org/abstract/document/10417860 (https://doi.org/10.1109/TCSI.2024.3355944)



###### 2.A Convolution Neural Network Accelerator Design with Weight Mapping and Pipeline Optimization

Paper: https://ieeexplore.ieee.org/abstract/document/10247977 (https://doi.org/10.1109/DAC56929.2023.10247977)



###### 3.A 29.12 TOPS/W and 1.13 TOPS/mm2 NAS-Optimized Mixed-Precision DNN Accelerator with Vector Split- andCombination Systolic in 28nm CMOS

Paper: https://ieeexplore.ieee.org/document/10529039 (https://doi.org/10.1109/CICC60959.2024.10529039)



###### 4.Advances and Trends on On-Chip Compute-in-Memory Macros and Accelerators

Paper: https://ieeexplore.ieee.org/abstract/document/10248014 (https://doi.org/10.1109/DAC56929.2023.10248014)



###### 5.A Nonvolatile AI-Edge Processor with 4MB SLC-MLC Hybrid-Mode ReRAM Compute-in-Memory Macro and 51.4-251TOPS/W

Paper: A Nonvolatile AI-Edge Processor with 4MB SLC-MLC Hybrid-Mode ReRAM Compute-in-Memory Macro and 51.4-251TOPS/W (https://doi.org/10.1109/ISSCC42615.2023.10067610)



###### 6.A 2.75-to-75.9TOPS/W Computing-in-Memory NN Processor Supporting Set-Associate Block-Wise Zero Skipping and Ping-Pong CIM with Simultaneous Computation and Weight Updating

Paper: https://ieeexplore.ieee.org/abstract/document/9365958 (https://doi.org/10.1109/ISSCC42613.2021.9365958)



###### 7.TensorCIM: Digital Computing-in-Memory Tensor Processor With Multichip-Module-Based Architecture for Beyond-NN Acceleration

Paper: https://ieeexplore.ieee.org/abstract/document/10555118 (https://doi.org/10.1109/JSSC.2024.3406569)



###### 8.A 1ynm 1.25V 8Gb, 16Gb/s/pin GDDR6-based Accelerator-inMemory supporting 1TFLOPS MAC Operation and Various Activation Functions for Deep-Learning Applications

Paper: https://ieeexplore.ieee.org/document/9731711 (https://doi.org/10.1109/ISSCC42614.2022.9731711)



###### 9.Unified Agile Accuracy Assessment in Computing-in-Memory Neural Accelerators by Layerwise Dynamical Isometry

Paper: https://ieeexplore.ieee.org/abstract/document/10247782 (https://doi.org/10.1109/DAC56929.2023.10247782)



###### 10.Processing-in-SRAM Acceleration for Ultra-Low Power Visual 3D Perception

Paper: https://dl.acm.org/doi/abs/10.1145/3489517.3530446 (https://doi.org/10.1145/3489517.3530446)
