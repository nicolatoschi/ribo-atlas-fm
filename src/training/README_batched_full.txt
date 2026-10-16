FIXED-WORK FULL-TRAJECTORY CONSTRUCT BATCHING
Actual public RNA3DB-pretrained RNAbpFlow inverse model,17,207,065trainable
parameters plus4673randomfrozen geometry/chemistry systems-head parameters.
Real archived1l8v_B157nt nativefeatures;32synthetic-observation caseinstances
of this oneRNA, not32independent experimentalstructures. K_train32,50native
modelcalls/trajectory(49Eulerintervals+finalcall). Finalgenerated profiles
remain Torch-differentiable, non-reentrant checkpointed, Kstreamed with
common-RNG two-pass mean-MSE gradients and native adjacent-distance term.
Samebulk/adjacency/teacher objectiveweights1/.1/1. Perconstruct weightedMSE
normalized separately then averaged, microgradients dividedbyaccumulation.

Fixedglobal32cases on4GPUs; eachrank8logicalcases. Micro2/4/8 accum4/2/1,
so exposure and objective are heldconstant. Same logical-case synthetic
probe/chemistry/seed acrossK andmicrobatch. Initialpriors seeded perlogical
case andmember beforeconcatenation. Teacher inputs t/trans_t/rotmats_t are
prepared once onCPU as singleton nativecorruptions and hash-frozen; they
are identical acrossmicrobatch, avoiding repeatedOT/Kabsch numericvariation.
Native teacher train-mode dropout distribution is unchanged but the RNG
layout depends on batching. This is disclosed; no identical full-teacher
gradient claim acrosslayouts. Case-mean reduction is the unchanged percase
objective ofthe single-case benchmark, averaged acrosscaseinstances.

ActualCPUproof: real75nt1g59_B, fixed2cases,K2/T2,micro1vs2 withteacherweight0
so generated+adjacency gradients canbe compared independentlyofteacher
dropout. MaximumFP32 parameter-gradientdifference5.70901e-7; allparameter
allcloseatol1e-5/rtol1e-4; replay0; generatedinverse/probegradientspositive
andfinite; frozenheadgradientsabsent. Separateactualbatch2T2 teacherweight1
fullupdate alsofinite. CPUproof doesnotestablishGPUbatch8/T50 performance.

EachGPU experimentmeasuresoneinitialoptimizerupdate, includingmeanpass,
gradientreplay/checkpointbackward/native teacher/DDP/finiteguards/clip/AdamW.
PeakHBMmustbe below85%capacity. Finalteacherbackward ofthe finalmicrobatch
reduces accumulatedallparametergradients once; earliermicrobatches no_sync.
Countersare actualmodelcallentries, notreplacementcostmultipliers. GPU
throughput/memory measuredbeforechoosingproductionmicrobatch; serialmicro1
cost isnotassumed necessary for85,500node-hour capacityforecast.

Original full860 and all submitted failures remain immutable. GPLupstream
licence retained. Syntheticobservations/randomhead establishexecution and
cost only; scientific forwardhead fitting and experimentalbenefit are separate.
