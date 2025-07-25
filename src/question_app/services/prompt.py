import os
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import jinja2
import jinja2.meta


class PromptifulError(Exception):
    pass


class TemplateLoadError(PromptifulError):
    pass


class TemplateRenderError(PromptifulError):
    pass


class Template(metaclass=ABCMeta):
    @abstractmethod
    def varnames(self) -> Collection[str]: ...
    @abstractmethod
    def _render(self, vals: Mapping[str, Any]) -> str: ...

    def render(self, vals: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
        if vals is None:
            vals = kwargs
        for key in self.varnames():
            if key not in vals:
                raise TemplateRenderError(f"variable {key!r} is missing in args")
        for key in vals.keys():
            if key not in self.varnames():
                raise TemplateRenderError(f"variable {key!r} is unknown to template")
        try:
            return self._render(vals)
        except Exception as exc:
            raise TemplateRenderError() from exc


class TemplateManager(metaclass=ABCMeta):
    @abstractmethod
    def _load_template(self, name: str) -> Template: ...

    def load_template(self, name: str) -> Template:
        try:
            return self._load_template(name)
        except Exception as exc:
            raise TemplateLoadError() from exc


class FsTemplateManager(TemplateManager):
    _priv_prefix = "_"

    @classmethod
    def _load_paths(cls, path: str | os.PathLike[str]) -> dict[str, Path]:
        root = Path(path).resolve()
        fpaths: list[Path] = []
        for bpath, dnames, fnames in os.walk(path):
            dnames[:] = [it for it in dnames if not it.startswith(cls._priv_prefix)]
            fpaths.extend(Path(bpath) / it for it in fnames if not it.startswith(cls._priv_prefix))
        name_to_path: dict[str, Path] = {}
        for fpath in fpaths:
            name = str(fpath.relative_to(root).with_suffix(""))
            if os.sep != "/":
                name = name.replace(os.sep, "/")
            if name in name_to_path.keys():
                raise TemplateLoadError(f"{name_to_path[name]!r} and {fpath!r} share the same path stem")
            name_to_path[name] = fpath
        return name_to_path


class JinjaTemplate(Template):
    def __init__(self, tmpl: jinja2.Template, varnames: set[str]) -> None:
        self._tmpl = tmpl
        self._varnames = varnames

    def varnames(self) -> set[str]:
        return self._varnames

    def _render(self, vals: Mapping[str, Any]) -> str:
        return self._tmpl.render(vals)


class JinjaTemplateManager(FsTemplateManager):
    def __init__(self, root: str | os.PathLike[str], **env_kwargs: Any) -> None:
        self._root = Path(root).resolve()
        self._name_to_path = self.__class__._load_paths(self._root)
        self._env = jinja2.Environment(loader=jinja2.FileSystemLoader(self._root), **env_kwargs)

    def _load_template(self, name: str) -> JinjaTemplate:
        path = self._name_to_path[name]
        load_name = str(path.relative_to(self._root))
        if os.sep != "/":
            load_name = load_name.replace(os.sep, "/")
        varnames = jinja2.meta.find_undeclared_variables(
            self._env.parse(path.read_text(encoding="utf-8"), load_name, path.name)
        )
        return JinjaTemplate(self._env.get_template(load_name), varnames)
