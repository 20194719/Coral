"""Oracle input and pair-judgment adapter without language-model requests for BioDEX dataset only. """

from pathlib import Path
from src.data.storage import read_pickle


def oracle_files(directory, view):
    """Locate oracle annotations within a dataset folder."""
    directory = Path(directory)
    paths = {
        key: directory / view["oracle_directory"] / f"{key}.pkl"
        for key in ("theta_library", "filter_library")
    }
    return paths


def validate_oracle_files(directory, views):
    """Check dataset annotations before starting model work."""
    files = sorted(
        {path for view in views for path in oracle_files(directory, view).values()}
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "BioDEX dataset is missing oracle files:\n" + "\n".join(missing)
        )
    return files


class Oracle:
    def inputs(self, directory, view, tables):
        """Prepare dataset judgment inputs."""
        paths = oracle_files(directory, view)
        paths["dataset_directory"] = Path(directory)
        paths["relation"] = view["relation"]
        tlib, flib = read_pickle(paths["theta_library"]), read_pickle(
            paths["filter_library"]
        )
        if (tlib["n_R"], tlib["n_S"]) != tuple(map(len, tables)):
            raise ValueError("Oracle dimensions do not match submitted tables")
        item = tlib["thetas"][view["relation"]]
        paths["condition"] = item["text"]
        positives = {(int(r), int(s)) for r, s in item["positives"]}
        members = {
            table.name: {
                text: set(value["members"]) for text, value in flib[side].items()
            }
            for side, table in zip(("phi_R", "phi_S"), tables)
        }
        return paths, positives, members

    def probe(self, view, condition, filters):
        """Build an online judgment function."""
        members = [
            view["members"][t.name][filters[t.name]] if t.name in filters else None
            for t in view["tables"]
        ]
        positives = view["positives"]

        def judge(pairs, requested):
            """Read requested oracle labels."""
            return [
                (
                    float((int(r), int(s)) in positives) if need[0] else None,
                    float(r in members[0]) if need[1] else None,
                    float(s in members[1]) if need[2] else None,
                )
                for (r, s), need in zip(pairs, requested)
            ]

        return judge
