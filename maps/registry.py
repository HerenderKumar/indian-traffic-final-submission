# maps/registry.py
import yaml
from pathlib import Path
from maps.map_config import MapConfig

class MapRegistry:
    _config_dir: Path = Path("maps/")

    @classmethod
    def load(cls, city_name: str) -> MapConfig:
        yaml_path = cls._config_dir / f"{city_name}.yaml"
        if not yaml_path.exists():
            available = [p.stem for p in cls._config_dir.glob("*.yaml")
                         if not p.stem.endswith("_festivals")]
            raise ValueError(f"No MapConfig for '{city_name}'. Available: {available}")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return MapConfig(**data)

    @classmethod
    def available_cities(cls) -> list:
        return [p.stem for p in cls._config_dir.glob("*.yaml")
                if not p.stem.endswith("_festivals")]