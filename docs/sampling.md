### PDF (probability density function)

This is a implementation of the [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling). 
It is done vi following algorythm:

1. `num_jets_estimate` are binned for each flavour using the configurations for resampling variable bins. This histogram is the initial estimate of the pdf of jets of each flavour.
2. The importance function is estimated using the ratio of the histograms `pdf_target_flavour/pdf_resampled_flavour` for each flavor except the target. Safe divesion is used i.e. if `pdf_resampled_flavour` is 0 the expressin defaults thus the jets where `pdf_target_flavour` or `pdf_resampled_flavour` is estimated by 0 are not resampled. 
3. Optionally the importance function is upscaled i.e. interpolated using cubic spline interpolation to a finer grid of bins. Centres of bins are used as nodes for the splines and the function is evaluated in the centeres of the new bins where the new bins are created by splitting the old bins in `upscale_pdf` intervals of equal width. This way the edge bins of each binning region are actually extrapolated rather than interpolated.
4. The new batch of jets is being read and after the cuts are applied `n_batch` jets remain. The jets are binned with the the binning from step 1 (if upscaling is not used) or upscaled binning defined by 3 (if upscaling is used) and the reference number of the bin for each jet is saved.
5. Each jet is assigned an importance score equal to the value of the importance function in the corresponding bin.
6. `n_batch*flavour.sampling_fraction` jets are selected with replacement using importances as weights.

This algryhtm is used for all the flavours except the target flavour for which all jets are saved without sampling as they already follow the desired distribution. One has to remember that `flavour.sampling_fraction==1` will lead to many jets beig selected more then once, choosing lower `sampling_fractions` can help agains it.

### Countup

Countup resampling tries to select as many unique jets from each bin as possible before selecting the duplicates.

1. `num_jets_estimate` are binned for each flavour using the configurations for resampling variable bins. This histogram is the initial estimate of the pdf of jets of each flavour.
2. The new batch of jets is being read and after the cuts are applied `n_batch` jets remain. The jets are binned with the the binning from step 1 (if upscaling is not used) or upscaled binning defined by 3 (if upscaling is used) and the reference number of the bin for each jet is saved.
3. The number of **requested** jets in each bin are calculated as `floor(n_batch*pdf_target_flavour+uniform([0, 1]))` so that if `n_batch*flavour.sampling_fraction*pdf_target_flavour=1.2` it has a 80% chance to be rounded up to 1 and 20% cnahce to be rounded up to 2 so that for each bin we get an integer number that on average corresponds to the expected value. 
4. From each bin we select consecutively (without replacement) the required number of jets. If the bin holds less jets than the **requested** number the rest of jets in this bin is chosen at random from this bin with replacemene. This way only few jets in each bin are repeated for `flavour.sampling_fraction=1` and rarely any are repeated for smaller sampling fractions
5. It may happen that we required jets from the bin that is empty in this batch thus the operations above lead to less jets than **requested** in total. To compencate we resample this number at random (with replacement) from already sampled jets. This leads to some additional repetitions but this way we can be sure that we adhere to the target pdf.


