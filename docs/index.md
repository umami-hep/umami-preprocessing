# Umami PreProcessing 

This is a modular preprocessing pipeline for jet tagging.

Training ntuples are produced using the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper) which dumps them directly into hdf5 files. The finished ntuples are also listed in the tables in the FTAG documentation section [here](https://ftag.docs.cern.ch/software/samples/). However, the training ntuples are not yet optimal for training the different _b_-taggers and require preprocessing.

This library is alredy used to preprocess data for [Salt](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/) framework.
UPP is planned to be integrated into [Umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami) framework for training of Umami/DIPS and DL1r and replace current umami preprocessing, as it addresses [several issues](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues/?label_name%5B%5D=Preprocessing) with the current umami preprocessing workflow, and uses the [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools/) package extensively.

## Motivation
The motivation for preprocessing the training samples results from the fact that the input datasets are highly imbalanced in their flavour composition. While there are large quantities of light jets, the fraction of _b_-jets is small and the fraction of other flavours is even smaller.
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more samples from the minority class (over-sampling). 

The resampling not only aims to have the same number of jets of each flavout but also to make the distributions of the kinematic variables like $p_T$ and $\eta$ same for all flavours. This is required to ensure similar kinematic distributions for the jets of different flavours in the training samples in order to avoid kinematic biases.

## Hybrid Samples
Umami/DIPS and DL1r are trained on so-called hybrid samples which are created using both $t\bar{t}$ and $Z'$ input jets.
The hybrid samples for PFlow jets are created by combining events from $t\bar{t}$ and $Z'$ samples based on a pt threshold, which is defined by the `pt_btagJes` variable for all jet-flavours.
Below a certain pt threshold (which needs to be defined for the preprocessing), $t\bar{t}$ events are used in the hybrid sample. Above this pt threshold, the jets are taken from $Z'$ events.
The advantage of these hybrid samples is the availability of sufficient jets with high pt, as the $t\bar{t}$ samples typically have lower-pt jets than those jets from the $Z'$ sample.

The following image show the distributions of the jet flavours in both components

![Pt distribution of hybrid samples being composed from ttbar and Zjets samples](assets/pt_btagJes-cut_spectrum.png)

After applying `pdf` resampling with upscaled function we achive the following combined distributions for jets:

![pT distribution of downsampled hybrid samples](assets/train_pt_btagJes.png)

Although we are using here for example reasons $t\bar{t}$ and $Z'$ to create a hybrid sample, you can use any kind of samples. Also, you don't need to create a hybrid sample. You can still use only one sample and for the preprocessing.





