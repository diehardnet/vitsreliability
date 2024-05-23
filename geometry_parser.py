import enum

import numpy as np


class ErrorGeometry(enum.Enum):
    MASKED = -1
    SINGLE = 0
    RANDOM = 1
    LINE = 2
    BLOCK = 3
    CUBIC = 4

    def __gt__(self, other): return self.value > other.value

    def __lt__(self, other): return self.value < other.value

    def __str__(self): return self.name

    def __repr__(self): return self.name


def geometry_comparison(diff: np.ndarray) -> ErrorGeometry:
    """
    This should receive as input a diff matrix np array with the same size of the tensors
    """
    count_non_zero_diff = np.count_nonzero(diff)
    if count_non_zero_diff == 1:
        return ErrorGeometry.SINGLE
    elif count_non_zero_diff > 1:
        dim = diff.ndim
        if dim > 3 or dim < 1:
            raise ValueError("Diff dimensions should be between 1-3")

        # Use label function to labeling the matrix
        where_is_corrupted = np.argwhere(diff != 0)

        # Get all positions of X and Y
        # Count how many times each value is in the list
        _, counter_x_positions = np.unique(
            where_is_corrupted[:, 0], return_counts=True
        )
        # Check if any value is in the list more than one time
        row_error = np.any(counter_x_positions > 1)

        col_error, box_error = None, None
        # Do the same for the other dimensions
        if dim >= 2:
            _, counter_y_positions = np.unique(
                where_is_corrupted[:, 1], return_counts=True
            )
            col_error = np.any(counter_y_positions > 1)

        if dim == 3:
            _, counter_z_positions = np.unique(
                where_is_corrupted[:, 2], return_counts=True
            )
            box_error = np.any(counter_z_positions > 1)

        dim_err_count = sum([1 for x in [row_error, col_error, box_error] if x])
        # print(dim, row_error, col_error, box_error)
        # print(np.nonzero(diff))

        if dim_err_count == 3:
            return ErrorGeometry.CUBIC
        # if row_error and col_error:  # square error
        elif dim_err_count == 2:
            return ErrorGeometry.BLOCK
        # elif row_error or col_error:  # row/col error
        elif dim_err_count == 1:
            return ErrorGeometry.LINE
        else:  # random error
            return ErrorGeometry.RANDOM

    return ErrorGeometry.MASKED

# BRUNO APPROACH
# def include_dimension(error_format, dim):
#     return f"{error_format} ({dim}D)"
#
#
# def current_argwhere():
#     # honestly, I have no idea when they changed numpy's argwhere signature
#     # however, the TX2 can only run up to python 3.6.9
#     # so if we are not running on the TX2, we (probably) have the current/updated
#     # np.argwhere signature
#     return sys.version_info.major >= 3 and sys.version_info.minor >= 8

# def geometry_comparison(diff, dim=2):
#     assert 1 <= dim <= 3, f"Geometry comparison only accepts 1D, 2D, or 3D tensors, got {dim}D tensor"
#     error_format = MASKED
#     count_non_zero_diff = np.count_nonzero(diff)
#     if count_non_zero_diff == 1:
#         error_format = SINGLE
#     elif count_non_zero_diff > 1:
#         # Use label function to labelling the matrix
#         where_is_corrupted = np.argwhere(diff != 0)
#
#         # Get all positions of X and Y
#         if current_argwhere():
#             all_x_positions = where_is_corrupted[:, 0]
#             if dim > 1:
#                 all_y_positions = where_is_corrupted[:, 1]
#         else:
#             all_x_positions = where_is_corrupted[0, :]
#             if dim > 1:
#                 all_y_positions = where_is_corrupted[1, :]
#         # print(all_x_positions)
#
#         # Count how many times each value is in the list
#         unique_elements, counter_x_positions = np.unique(all_x_positions, return_counts=True)
#         # print(counter_x_positions)
#         if dim > 1:
#             unique_elements, counter_y_positions = np.unique(all_y_positions, return_counts=True)
#         # exit(0)
#
#         # Check if any value is in the list more than one time
#         row_error = np.any(counter_x_positions > 1)
#
#         col_error = False
#         if dim > 1:
#             col_error = np.any(counter_y_positions > 1)
#
#         # Only do the same for Z if dim = 3
#         depth_error = False
#         if dim == 3:
#             if current_argwhere():
#                 all_z_positions = where_is_corrupted[:, 2]
#             else:
#                 all_z_positions = where_is_corrupted[2, :]
#
#             unique_elements, counter_z_positions = np.unique(all_z_positions, return_counts=True)
#             depth_error = np.any(counter_z_positions > 1)
#
#         dimm_err_count = sum([1 for x in [row_error, col_error, depth_error] if x])
#
#         if dimm_err_count == 3:
#             error_format = CUBE
#         # if row_error and col_error:  # square error
#         elif dimm_err_count == 2:
#             error_format = SQUARE
#         # elif row_error or col_error:  # row/col error
#         elif dimm_err_count == 1:
#             error_format = LINE
#         else:  # random error
#             error_format = RANDOM
#     return include_dimension(error_format, dim)
