
# Abstract 

This paper here:

* tries to make an SRAM-style all-digital CIM 
* shows some ways it fails
* Shows some alternatives, and their problems
* Shows the power efficiency of a few of its constituent parts, without the SRAM

See [@rekhi2019] or [@chih2021] or [see @kim2019 p.26] for examples of citing stuff here.  



# Introduction 

Increased demand for high-performance neural-network training and inference has driven a wide array of machine-learning acceleration hardware. Typical such ML accelerators feature large data-parallel arithmetic hardware arrays, such as those for performing rapid dense-matrix multiplication. Such accelerators frequently include more arithmetic capacity than their attached memory systems can supply, rendering them heavily memory-bound. Common tactics for confronting these memory-bandwidth limitations have included ever-larger local caches, often dominating an accelerator's overall die area. 
As machine-learning acceleration occurs in both high-performance and low-power contexts (i.e. on mobile edge devices), the speed and energy-efficiency of these operations is of great interest. 

Two commonly linked, although conceptually separable, classes of such accelerators have risen to prominence in recent research: those utilizing analog signal processing, and those utilizing "compute-in-memory" circuits which break the classical Von Neumann split between processing and memory-storage, incorporating logic functions among memory arrays. 



## Analog Signal Processing

Analog methods for neural-network acceleration, particularly that of matrix-matrix multiplication, commonly deploy analog-domain processing for one or both of their MACC's attendant arithmetic operations: multiplication and/or addition. Analog multiplication is typically performed via a physical device characteristic, e.g. transistor voltage-current transfer [@chen2021], or that of an advanced memory cell such as RRAM[@yoon2021] or ReRAM [@xue2021]. Addition and accumulation are most commonly performed either on charge or current, the two analog quantities which tend to sum most straightforwardly. More elaborate systems use time-domain signal-processing [@sayal2020] for their arithmetic operations. 

In principle these analog-domain operations can be performed at both high speed and high energy-efficiency. Their primary cost, somewhat paradoxically, is a *reduction* in fidelity. Despite representing each of their values in functionally infinite-resolution physical quantities, analog processing commonly forces signals to be quantized to just a few bits. As the native format of both upstream and downstream processing is digital, these accelerators require a domain-conversion of both their inputs (via DACs) and outputs (via ADCs). Resolution and performance of these converters is a material design constraint. This largely drives the analog techniques' second pitfall: their substantially higher design cost, both for the data converters and core arithmetic cells. 

Prior research [@rekhi2019] has set an upper bound on the resolution-efficiency trade-off for such analog-computation accelerators, by (a) presuming the analog-to-digital conversion as the energy-limiting step, and (b) assuming state-of-the-art ADC performance and efficiency. But this bound is likely far too permissive. Such accelerators obviously (a) have other energy-consuming elements besides their ADCs, but more importantly, (b) do not necessarily (or even likely) feature appropriate trade-offs for state-of-the-art data conversion. Such converters often consume substantial die area, and/or require elaborate calibration highly attuned to their use-cases. To the author's knowledge no research-based attempts have been made to capture the performance of converters used in such accelerators relative to the state-of-the-art. 

Furthermore, a substantial complication is altogether ignored: analog computation is inherently non-deterministic. Analog signals and circuits have irrecoverable sources of thermal, flicker, and shot noise, which can only be budgeted against, but never fully removed. Some proportion of the time, an analog multiplier will inevitably report that 5 x 5 makes 26, or 24: the designer's only available knob is *how often*. This is equivalent to choosing a thermal-noise SNR, a process widely understood in data-conversion literature to *quadruple* power consumption per added bit. 

The analysis presented by Rekhi implicitly buckets all "AMS Error" as that commonly called *quantization error* or *quantization noise* in data-conversion space. While often fairly intractable across a large batch of devices, these errors are deterministic for a given device once fabricated, while held under unvarying conditions. Thermal noise, in contrast, varies per operation, including between multiplications in the same inference cycle. To the author's knowledge no treatment of such inherently-random error sources exists in the literature on these analog techniques. While a category of *stochastic neural networks* attempts to intentionally include such noise sources, these techniques are not what popular networks or accelerators thereof attempt to compute. 



## Compute-in-Memory 

Digital compute-in-memory circuits, in contrast, retain the digital-word representation of each quantity, and retain the boolean-logic implementation of their core arithmetic function. Their distinct characteristic is that said arithmetic is dispersed among a memory-and-compute array, the atoms of which are not generally available to common digital-design flows (synthesis, place-and-route layout). Digital CIM cells appear to their surrounding systems more similar to SRAM arrays which internally perform some amount of computation. The internal atoms of these circuits generally include (a) one or more high-density storage elements, similar to those of a typical SRAM bit-cell, and (b) a paired atom of computation, often a single-bit or small-resolution multiplication. Peripheral circuits include those similar to typical SRAMs, designed to read or write words at a time, as well as those designed for combining atomic-level operations, such as accumulation of partial products. Many such circuits are designed for wide flexibility among operand-types, often serializing smaller internal operands to form larger inputs and outputs.  [@chih2021] does so on four-bit units, while [@kim2019] and [@kim2021] extend this idea all the way to single-bit serial operation. 

The primary question for such digital array is whether they offer sufficient benefit to justify their design cost. Such circuits require leaving typical digital-design flows and adopting a custom design processes similar to that of SRAM, a field now commonly reserved for the largest and best-resourced industrial teams. (Although notably still less complex than a full-analog design process.) SRAM arrays however include a material constraint which CIM accelerators need not: SRAM bit-cells are generally designed to be instantiated billions of times over, and accordingly designed to incredibly high yield. Many such cells are designed to fail only when outside of six or seven standard-deviations worth of manufacturing tolerances. Even these low failure rates justify the overhead of redundancy and error-correction peripheral circuits. CIM accelerators, in contrast, are likely to include several orders of magnitude fewer bits, and may allow for correspondingly higher failure rate of their atomic-units. The circuit-level arrangement of these atomic-units can and does in many cases ease common SRAM-design constraints, such as the tension between bit-cell readability and writability. 

While CIM and analog-processing are commonly intertwined, we again note their conceptual separability. Analog matrix-multiply arrays typically incorporate local weight storage, hence their "in-memory" naming. But this weight-storage is typically tied to the size of the array, and much smaller than even modest local SRAM buffers. These "memories" are often more analogous in size and function to the register-based buffers distributed throughout a systolic array. Note that at no point in the preceding section's treatment of analog accelerators, nor in Rekhi's analysis, has the term "memory" been invoked at all.  

![concept](fig/cim-concept.pdf "Digital CIM Concept")

Refer to Figure~\ref{fig:concept} for the idea chief 


## Proposed Work 

Both compute-in-memory and analog computation are back-end implementation techniques for the same popular neural network operation: matrix-matrix multiplication. Evaluation of their effectiveness is primarily a back-end, physical-design activity, requiring relevant process technology information and relatively detailed design. This work will focus on these layers, and re-use existing research infrastructure for essentially all layers above them, primarily Gemmini and its software stack. This work will primarily include: 

* Energy and performance characterization of existing research-grade accelerator(s) (e.g. Gemmini) in relevant process technology(s)
* Design of a digital compute-in-memory macro similar to that of [@chih2021] and [@kim2019], and associated characterization and comparison of its area and energy-effectiveness 
* Time-Permitting: a modeling-based study of the effects of, and limitations imposed by, thermal-noise generation in analog-based accelerators, potentially further refining Rekhi et al's outer bound on their effectiveness 

Breaking this work into segments: the first will feature the back-end evaluation of the existing accelerator(s) in the target technology(s), and outline the design of the proposed digital accelerator. The second will feature the detailed design of the digital accelerator. And the third will detail the area and energy-efficiency achieved in its design efforts, as compared to those of the existing accelerator(s). 

Like most such pieces of detailed implementation work, these efforts will largely require targeted verification. Large integrated simulations of RTL-level processors running billions of instructions (a philosophical preference in the BAR community) would require impractical simulation times if utilizing transistor-level models of such compute-arrays. As is common in mixed-signal environments, any such full-system evaluation will use simplified but interface-compatible RTL models of these arrays, designed to capture their characteristics as visible to the larger system: their interfaces, behavior, and relavant timing delays. Comparison of these models against their implementations then occurs separately. 

This work will require the generation of several pieces of highly repetitive, high density custom-digital layout. Past UCB EECS work such as the Berkeley Analog Generator has designed methods for programmatic IC layout generation, but has focused on (a) analog circuits, and (b) their generality and process-portability to an extent incompatible with high-density digital layout, such as that common in SRAM. This work will instead utilize an in-progress framework for gridded, standard-cell-style semi-custom layout, and will serve as an early use-case for this framework. 


# Proposed All-Digital CIM Macro 

The proposed all-digital compute-in-memory macro is depicted in Figure~\ref{fig:macro}. Like @chih2021, @TBD, and many similar digital CIM desings, this work operates *bit-serially* on its input activations. Its primary memory-and-compute array is comprised of a set of weight columns, each of which includes an input-activation serializer, a parametric `NROWS` CIM "molecules" comprising eight-bit-word weight storage and compute, and a per-column reduction circuit comprising an adder tree and shift-and-accumulate multiplier. The macor stores and operates on weights directly in its CIM array, and includes additional shared SRAM for input activations, partial matrix-products, weight overflow, and any ancillary data. 

![macro](fig/cim.png "Proposed CIM Macro")

Figure~\ref{fig:column} shows each CIM column in greater detail. In each cycle, each column is broadcast a single bit of `NROWS` input activations. The column multiplies these `NROWS` bits by `NROWS` eight-bit weight-words, stored in its SRAM-based CIM atoms. These `NROWS` eight-bit products are then summed by a fully-combinational adder tree into an `NROWS + log2(NCOLS)`-bit partial product, which is shifted per its input-activation bit-weight, and accumulated onto past partial sums. Note despite its nominal function performs matrix-vector multiplies, it includes no traditional digital multiplier. Bit-serialized partial sums are instead shifted and accumulated into each column accumulator. For `NBITS`-width input activations, computing a matrix-vector product of dimensions `1 x NROWS` by `NROWS x NCOLS` to create an `NCOLS`-length row-vector requires `NBITS` clock cycles. 

![column](fig/cim-column.png "CIM Column")

The macro is designed to maximally simplify its primary unit, the CIM Atom or Bit-Cell. Figure~\ref{fig:atom} depicts the CIM atom circuit. It comprises a six-transistor SRAM bit-cell and an atomic *compute bit-cell*. Because input activations are bit-serialized, the width of each partial-product is equal to that of a weight-word, and multiplication can hence be performed with a single logic gate. Given appropriate polarity of weights and inputs, either of the primary universal gates NAND or NOR principally provide the necessary multiplication function. This work uses the minimum-sized NOR2 gate. 

Note each compute-bit operates directly its paired SRAM bit-cell's internal state, reaching around its access transistor. Weights are written but never read from the CIM SRAM bits; it operates as a sort of write-only SRAM. Its paired SRAM peripheral circuitry can in principle be reduced to reflect this use model, removing any read-circuits such as sense amplifiers and column muxes. 

![atom](fig/bitcell.pdf "CIM Atom, or Bit-Cell")

This was designed and compared against prior systolic-array-based research accelerator Gemmini (@genc2019gemmini), also developed at UC Berkeley, in TSMC 28nm and Skywater 130nm technologies. The latter's design-kit is provided as near-fully open-source software. (TBD: reference for this?) 

The `NROWS x NCOLS`-sized CIM and systolic arrays both produce a series of `NCOLS`-length vectors. The systolic array produces one such vector per cycle, while for `NBITS`-wide inputs, the CIM array throughput is reduced the same factor `NBITS`. The CIM array therefore produces a vector result every `NBITS` cycles. Defining an operation as a scalar weight-width multiplication or addition, the CIM arrays' operations per cycle equals that of the systolic array in the case in which its array-dimensions are `NBITS` times larger. For any larger dimensions, its ops/cycle are greater, and for any smaller dimensions they are less. In this work both the CIM and systolic array weight and input sizes (`NBITS`) are set to eight bits. While this requires a CIM array of at least 8 times as many elements, we note that each element in the systolic array includes substantially more hardware: an 8x8 bit hardware multiplier, larger-width accumulator, and registers for enacting the two-dimensional systolic pipelining. The CIM unit, in contrast, requires only a storage-compute molecule plus its share of the reduction logic, amortized over `NROWS` such molecules. 


## Atom vs Reduction Area 

The liabilities of the all-digital CIM proposed here, and of those proposed in recent research, lie in the design of their reduction function. 

In each of the two evaluation technologies, the minimum-sized standard logic gate is roughly 2x that of the single-port SRAM bit cell. Presuming that its widespread re-use has pushed the SRAM bit-cell to near its design-rule-derived minimum size, and that a single two-input logic gate can be similarly optimized onto minimum size, we plan for a CIM atom of roughly 2x the SRAM bit-cell area. Product-outputs from pairs of these CIM atoms are then fed into the inputs of a binary adder tree. Both evaluation technologies provide full-adder cells as part of their standard libraries; in both cases the area of such cells is roughly *12x* that of the SRAM bit-cell. At such proportions, the adder tree's input stage alone requires roughly 3x the area of the CIM atoms. While these cells might also be redesigned for minimal area, it remains unlikely that they would near that of the single-bit multiplier or SRAM bit-cell. The NAND/NOR realization of such a gate requires nine such cells; common XOR-based implementations commonly use in excess of 20 transistors. 

Initial VLSI-flow-based layout of 64-row CIM columns yielded adder-tree areas between 24x and 81x the SRAM area, depending on their target frequency. The relatively wide range illustrates a material constraint in the adder-tree design: the fastest such circuits generally trade area in exchange for speed. Carry-select is a prominent example, widely generated by logic synthesis; it roughly doubles the adder area by including two parallel adders, multiplexed to select the correct carry value. Low-area adders tend to be slow adders. The lowest-area known to the author is also the most basic: the ripple-carry. [@chih2021] makes explicit their use of such in the reduction tree. 

Figure TBD depicts the proportion of area dedicated to the reduction adder-tree for all relevant parametric combinations of `NBITS` and `NROWS`, using the min-area-architecture ripple adder. In each case the *quantity* of SRAM bit and full-adder unit-cells is a ratio between 1:1 and 1.4:1. The larger unit-area of the full-adder, here presumed to be 6x that of the bit-cell, renders its area contribution between 80 and 85 percent of the array. This estimate for the area ratio is subject to both upward and downward pressure, as (a) a custom FA unit-cell can be designed at lower area than the standard-library's, as in [@chih2021], but (b) arrangement of the binary-tree of these cells onto a rectangular array would likely incur significant overhead. 

Prior work such as [@chih2021] designs such a ripple-based adder tree, and a custom full-adder cell therefore. While no analysis is provided regarding its area breakdown, a total of 64Kb worth of 0.379µm2 bit-cells are included in an overall 202k µm2 macro, comprising roughly 12 percent of its overall area. We expect the remaining area and relatively low proportion of memory-area is, as in this work, due to the area demands of the adder tree. 

# Alternate Array Designs 

Some kinda intro to showing other ideas that might help

![altcolumn](fig/other-cim-column.png "Alternate CIM Column, similar to [@kim2021]")
![systolic](fig/systolic.pdf "Concept Systolic-in-Memory Array")

Figure~\ref{fig:altcolumn} depicts a concept CIM array in which the reduction tree is atomized into each bit-cell. The CIM atoms then comprise the prior SRAM bit-cell and multiplier, plus a newly-added full-adder cell. This arrangement is highly similar to that presented in [@kim2019] and [@kim2021]. Partial products are passed down the array, while carries are propagated across weight-words. The array retains the bit-serial input-activation application, and the per-column shift-and-accumulate multiplier. 

Such an array theoretically maximizes the proportion of its area dedicated to CIM atoms. However not all such atoms can be fully utilized. In [@kim2021], the row-ended adders are also comprised of CIM atomic cells. The weight-storing and add-only atoms are categorized as one of Types I or II. For the latter, multiplication is disabled. Per row, `log2(NROWS)` of these adder-only atoms are required. For eight-bit weights, this requires a 7/15 overhead. These overheads remain constant in size as weight-widths shrink; results running binary networks require a full 7/8 overhead. This can be visualized as: the sum of 128 eight-bit words will require `log2(128) + 8 = 15` bits. Rather than arrange the progressively increasing width in a tree, *each* row includes all 15 bits. Just as impactfully, the inclusion of the full-adder cell in the atom requires that *each* atom - rather than each *pair of atoms* - include such a cell. The binary-splitting of the tree-based design begins at its products; only each input-stage adder is functionally amortized across two rows. 
The total reduction function also includes a carry-path through most elements in the array, limiting is speed. Clock frequencies reported in [@kim2021] rise only to the tens of megahertz, despite being implemented in a 65nm process with gate-delays likely of a few dozen picoseconds. 

To alleviate such slow-downs, one might imagine a *systolic-in-memory array* such as that depicted in Figure~\ref{fig:systolic}. Such an array applies pipelining to one or both of its dimensions. Setting aside whether the relatively low-cost single-bit input activations are pipelined, we note that solely adding internal pipelining to the partial product accumulations incurs a substantial hardware overhead. Each atom must include its own adder and accumulator register. Like Figure~\ref{fig:altcolumn}, these accumulators must each be of width `NBITS + log2(NROWS)`. For typical array sizes, this width will be roughly twice `NBITS`, rendering roughly 2/3 of memory usage to internal partial sums. If SRAM is used to implement these accumulators, additional "micro writers" must be designed which perform SRAM writes on a single (accumulator-width) word at a time. While entirely achievable at the circuit-level, the physical design of these circuits may apply further pressure on the atom's ability to use the most-optimized SRAM cells. 

![offline](fig/reduction-offline.pdf "Concept Offline Reduction in SRAM Periphery")
![sparse](fig/near-mem-sparse.pdf "Near-Memory Array with Bit-Level Sparsity")

Lastly, we consider the prospects of removing *all* circuitry from the array, but for that of the atomic unit. Figure~\ref{fig:offline} depicts the concept of such an "offline reduction" array. Such operation requires that input activations be *doubly* serialized, both bit-wise and word-wise, as only a single row can be read at a time. The reduction function can then be performed in the array's peripheral circuitry. 

Examining such a design quickly produces the realization that it does not require a custom array at all. Each word read from the array of Figure~\ref{fig:offline} is one of two values: a weight-word, or zero. Further, (a) the peripheral circuit *knows* many of the zero-values, as it can examine the bit-serial input activations, and (b) zero-values are only accumulated, so we need not retrieve them at all. The combination of these realizations motivates Figure~\ref{fig:sparse}, which might more accurately be called a compute-*near*-memory accelerator. Note Figure~\ref{fig:offline} uses a standard SRAM, without any custom array or periphery. It adds three elements: (a) column shift-accumulator multipliers, similar to those reviewed throughout, (b) an input serializer, also reviewed throughout this work, and (c) the sole novel entry, a dispatch unit which reads weight-rows corresponding to non-zero bit-values of input-activations, and sends a corresponding stream of `shift` and `completion` packets to the accumulators. All three are compatible with a standard VLSI flow, requiring no SRAM-level design. Such a design capitalizes on the *bit-level sparsity* of input activations, performing material work solely for bits which are non-zero. Common neural network designs include many activation bits - and in fact many full activation *words* - which are equal to zero, largely due to non-linear rectification. 

This section's alternative designs are presented at the level of detail at which they have been investigated: the block-diagram level. While that of Figure~\ref{fig:sparse} is of particular future interest as a low-area, modest performance acceleration solution, the focus of this work remains on the proposed CIM array, to which we now return. 


# Energy Comparisons 

Tree vs systolic/ Gemmini, other peoples comps and mine, etc

# Conclusions 

* As conceived this aint great
* Power efficiency of combinational/ bit-serial stuff lookin great though 
* 

The author thanks the team at TSMC for access to its 28nm technology, and the teams at Skywater and Google for their efforts promoting the open-source design-kit for the 130nm technology. Particular thanks are due to A. Gonzalez and Professor S. Shao, both of UC Berkeley, for their invaluable review and input.  

