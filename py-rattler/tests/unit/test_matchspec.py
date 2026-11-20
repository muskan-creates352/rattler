from pathlib import Path

import pytest

from rattler import BuildString, MatchSpec, NamelessMatchSpec, PackageRecord, RepoDataRecord


def test_parse_channel_from_canonical_name() -> None:
    m = MatchSpec("conda-forge::python[version=3.9]")
    assert m.channel is not None
    assert m.channel.name == "conda-forge"
    assert m.channel.base_url == "https://conda.anaconda.org/conda-forge/"


def test_parse_channel_from_canonical_name_nameless() -> None:
    m = MatchSpec("conda-forge::python[version=3.9]")
    nms = NamelessMatchSpec.from_match_spec(m)
    assert nms.channel is not None
    assert nms.channel.name == "conda-forge"
    assert nms.channel.base_url == "https://conda.anaconda.org/conda-forge/"


def test_parse_channel_from_url() -> None:
    m = MatchSpec("https://conda.anaconda.org/conda-forge::python[version=3.9]")
    assert m.channel is not None
    assert m.channel.name == "conda-forge"
    assert m.channel.base_url == "https://conda.anaconda.org/conda-forge/"


def test_parse_channel_from_url_nameless() -> None:
    m = MatchSpec("https://conda.anaconda.org/conda-forge::python[version=3.9]")
    nms = NamelessMatchSpec.from_match_spec(m)
    assert nms.channel is not None
    assert nms.channel.name == "conda-forge"
    assert nms.channel.base_url == "https://conda.anaconda.org/conda-forge/"


def test_parse_channel_from_url_filesystem() -> None:
    m = MatchSpec("file:///Users/rattler/channel0::python[version=3.9]")
    assert m.channel is not None
    assert m.channel.name == "channel0"
    assert m.channel.base_url == "file:///Users/rattler/channel0/"


def test_parse_channel_from_url_filesystem_nameless() -> None:
    m = MatchSpec("file:///Users/rattler/channel0::python[version=3.9]")
    nms = NamelessMatchSpec.from_match_spec(m)
    assert nms.channel is not None
    assert nms.channel.name == "channel0"
    assert nms.channel.base_url == "file:///Users/rattler/channel0/"


def test_parse_channel_from_url_localhost() -> None:
    m = MatchSpec("http://localhost:8000/channel0::python[version=3.9]")
    assert m.channel is not None
    assert m.channel.name == "channel0"
    assert m.channel.base_url == "http://localhost:8000/channel0/"


def test_parse_channel_from_url_localhost_nameless() -> None:
    m = MatchSpec("http://localhost:8000/channel0::python[version=3.9]")
    nms = NamelessMatchSpec.from_match_spec(m)
    assert nms.channel is not None
    assert nms.channel.name == "channel0"
    assert nms.channel.base_url == "http://localhost:8000/channel0/"


def test_parse_no_channel() -> None:
    m = MatchSpec("python[version=3.9]")
    assert m.channel is None
    assert m.name is not None
    package_name = m.name.as_package_name()
    assert package_name is not None
    assert package_name.normalized == "python"
    assert m.version == "==3.9"


def test_parse_no_channel_nameless() -> None:
    m = MatchSpec("python[version=3.9]")
    nms = NamelessMatchSpec.from_match_spec(m)
    assert nms.channel is None
    assert nms.version == "==3.9"


def test_extras_parsing() -> None:
    """Test that extras syntax can be parsed."""
    m = MatchSpec("numpy[extras=[test]]")
    assert m.name is not None
    assert m.name.normalized == "numpy"
    assert m.extras == ["test"]


def test_extras_multiple() -> None:
    """Test parsing multiple extras."""
    m = MatchSpec("numpy[extras=[test,dev,docs]]")
    assert m.name is not None
    assert m.name.normalized == "numpy"
    assert m.extras == ["test", "dev", "docs"]


def test_conditionals_parsing() -> None:
    """Test that conditionals syntax can be parsed."""
    m = MatchSpec('requests[when="python>=3.6"]')
    assert m.name is not None
    assert m.name.normalized == "requests"
    assert m.condition == "python>=3.6"


def test_deprecated_if_syntax_returns_error() -> None:
    """Test that the deprecated '; if' syntax returns an error."""
    with pytest.raises(Exception):
        MatchSpec("requests; if python >=3.6")


def test_extras_and_conditionals_together() -> None:
    """Test using both extras and conditionals together."""
    m = MatchSpec('numpy[extras=[test], when="python>=3.7"]')
    assert m.name is not None
    assert m.name.normalized == "numpy"
    assert m.extras == ["test"]
    assert m.condition == "python>=3.7"


def test_nameless_extras() -> None:
    """Test that NamelessMatchSpec also supports extras."""
    nms = NamelessMatchSpec(">=1.20[extras=[test]]")
    assert nms.version == ">=1.20"
    assert nms.extras == ["test"]


def test_strict_mode_with_extras() -> None:
    """Test that extras work with strict parsing mode."""
    m = MatchSpec("numpy[extras=[test]]", strict=True)
    assert m.name is not None
    assert m.name.normalized == "numpy"
    assert m.extras == ["test"]


def test_lenient_mode_with_extras() -> None:
    """Test that extras work with lenient parsing mode."""
    m = MatchSpec("numpy[extras=[test]]", strict=False)
    assert m.name is not None
    assert m.name.normalized == "numpy"
    assert m.extras == ["test"]


def test_extras_and_condition_none_by_default() -> None:
    """Test that extras and condition are None when features are not enabled."""
    m = MatchSpec("numpy >=1.20")
    assert m.name is not None
    assert m.name.normalized == "numpy"
    assert m.version == ">=1.20"
    assert m.extras is None
    assert m.condition is None


def test_matches_subdir() -> None:
    record = PackageRecord(name="foo", version="1.0", build=BuildString("0"), build_number=0, subdir="linux-64")
    spec_matching = MatchSpec("foo[subdir=linux-64]")
    spec_mismatch = MatchSpec("foo[subdir=win-64]")
    assert spec_matching.matches(record)
    assert not spec_mismatch.matches(record)

    nms_matching = NamelessMatchSpec("1.0[subdir=linux-64]")
    nms_mismatch = NamelessMatchSpec("1.0[subdir=win-64]")
    assert nms_matching.matches(record)
    assert not nms_mismatch.matches(record)


@pytest.mark.asyncio
async def test_matches_repodata_record(test_data_dir: str) -> None:
    path = Path(test_data_dir) / "clobber" / "clobber-fd-1-0.1.0-h4616a5c_0.conda"
    record = await RepoDataRecord.from_package_archive(path)
    spec_exact_fn = MatchSpec('clobber-fd-1[fn="clobber-fd-1-0.1.0-h4616a5c_0.conda"]')
    spec_diff_fn = MatchSpec('clobber-fd-1[fn="clobber-fd-1-0.1.0-h4616a5c_0.tar.bz2"]')
    assert spec_exact_fn.matches(record)
    assert not spec_diff_fn.matches(record)
