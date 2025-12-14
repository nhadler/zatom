from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS

# Keep only those groups acting on R^3 (not R^2)
PLATONIC_GROUPS_3D = {name: group for name, group in PLATONIC_GROUPS.items() if group.dim == 3}
