# Resampling
There are two main strategies for handling the issue of different kinematic properties for different jet flavours. These are reweighting and resampling. Reweighting applies a weight to each data point when calculating the loss, resulting in a weighted average loss for the considered data points. Resampling changes the actual data distribution by over- or undersampling data to achieve the desired data distribution. Statistically, the two methods are equivalent (in expectation). However, [in some cases](https://web.stanford.edu/~lexing/resw.pdf), resampling may result in a more numerically stable approximation of the minima of the loss function. Empirically, it has been found that resampling produces better results for jet flavour tagging performance.

For resampling, UPP has two different methods implemented. The desired method (`pdf` or `countup`) can be specified in the configuration file.

### Skipping resampling

Resampling can be disabled entirely by either omitting the `resampling` block from the config or setting `method: none`. In this case no `target`, resampling `variables`, or histogram (`--prep`) step are required. The jets passing the cuts are written directly, capped at each component's `num_global_objects`. Setting `num_global_objects: -1` (also valid for `num_global_objects_val` / `num_global_objects_test`) writes **all** jets of that component passing the cuts.

Note that the `--no-resample` command line flag is different: it only skips the resampling *stage* (for example to re-run the merge/norm/plot stages on existing component files) and does not disable resampling.

### PDF (probability density function)

This is an implementation of an [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) method. The aim of the algorithm is to ensure that the kinematic probability density functions (pdfs) of all jet flavours are matched to that of a chosen target flavour.

The resampling is done using the following steps:

1. A `num_global_objects_estimate` number of jets are binned for each flavour using the configurations for resampling variable bins. This histogram, `pdf_resampled_flavour`, is the initial estimate of the pdf of jets of each flavour.
2. The importance function is estimated by using the ratio of the histograms for each flavor to that of the target flavour, `pdf_target_flavour/pdf_resampled_flavour`. Safe division is used, which ensures that if for a bin in `pdf_resampled_flavour` is 0, we skip that bin. This ensures that we do not divide by 0. If a bin in `pdf_target_flavour` is 0, we also skip the bin. 
3. Optionally, the importance function is upscaled. This means that it is interpolated using cubic spline interpolation to a finer grid of bins. The centres of bins are used as nodes for the splines. The new bins are created by splitting the old bins into `upscale_pdf` number of bins of equal width. The function is evaluated in the centers of the new bins. This way, the edge bins of each binning region are actually extrapolated rather than interpolated.
4. The new batch of jets is being read and after the cuts are applied `n_batch` jets remain. The jets are binned with the the binning from step 1 (if upscaling is not used) or upscaled binning defined by 3 (if upscaling is used) and the reference number of the bin for each jet is saved.
5. Each jet is assigned an importance score equal to the value of the importance function in the corresponding bin.
6. `n_batch*flavour.sampling_fraction` jets are selected with replacement using importance scores as weights.

This algorithm is used for all the flavours except the target flavour for which all jets are saved without sampling as they already follow the desired distribution. One has to remember that `flavour.sampling_fraction==1` will lead to many jets being selected more then once, choosing lower `sampling_fractions` can help against it.

### Countup

Countup resampling tries to select as many unique jets from each bin as possible before selecting the duplicates.

1. `num_global_objects_estimate` jets are binned for each flavour using the configurations for resampling variable bins. This histogram is the initial estimate of the pdf of jets of each flavour.
2. The new batch of jets is being read and after the cuts are applied `n_batch` jets remain. The jets are binned with the the binning from step 1 (if upscaling is not used) or upscaled binning defined by 3 (if upscaling is used) and the reference number of the bin for each jet is saved.
3. The number of **requested** jets in each bin are calculated as `floor(n_batch*pdf_target_flavour+uniform([0, 1]))` so that if `n_batch*flavour.sampling_fraction*pdf_target_flavour=1.2` it has a 80% chance to be rounded up to 1 and 20% chance to be rounded up to 2 so that for each bin we get an integer number that on average corresponds to the expected value. 
4. From each bin we select consecutively (without replacement) the required number of jets. If the bin holds less jets than the **requested** number the rest of jets in this bin is chosen at random from this bin with replacement. This way only few jets in each bin are repeated for `flavour.sampling_fraction=1` and rarely any are repeated for smaller sampling fractions
5. It may happen that we required jets from the bin that is empty in this batch thus the operations above lead to less jets than **requested** in total. To compensate we resample this number at random (with replacement) from already sampled jets. This leads to some additional repetitions but this way we can be sure that we adhere to the target pdf.


