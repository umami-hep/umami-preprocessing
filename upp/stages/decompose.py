from pathlib import Path
from ftag import Transform
from ftag.hdf5 import H5Reader

from upp.classes.components import Components
from upp.classes.variable_config import VariableConfig

def decompose(
    components: Components,
    output_directory: Path,
    output_variables: VariableConfig,
    batch_size: int,
    input_jets_name: str,
    input_transform: Transform | None,
):
    for sample, sample_components in components.groupby_sample():
        # TODO: sample i/o should be handled by the sample class/module
        reader = H5Reader(
            list(sample.path),
            batch_size=batch_size,
            jets_name=input_jets_name,
            transform=input_transform,
            # TODO: this shouldn't be part of the Component class,
            # this should be a part of the Sample class
            equal_jets=sample_components.equal_jets
        )

        for region, region_components in sample_components.groupby_region():
            region_components.setup_writers(
                output_directory,
                output_variables,
                reader.dtypes(output_variables.combined()),
                lambda component: reader.shapes(
                    component.num_jets, list(output_variables.keys())
                )
            )

            # we need the additionally read the variables that we need to apply the cuts
            # TODO: why does the user need to think about this? This should be fixed in
            # the ftag package
            variables_to_read = output_variables.add_jet_vars(
                region_components.cuts.variables
            ).combined()
            for jet_batch in reader.stream(variables_to_read, cuts=region.cuts):
                region_components.write(jet_batch)
