import common
import configs
import dnn_log_helper
from setup_grounding_dino import run_setup_grounding_dino
from setup_selective_ecc import run_setup_selective_ecc


def main():
    args, args_text_list = common.parse_args()

    if args.code_type == configs.GROUDING_DINO:
        run_setup_grounding_dino(args=args, args_text_list=args_text_list)
    elif args.code_type == configs.MAXIMALS:
        pass
    elif args.code_type == configs.SELECTIVE_ECC:
        run_setup_selective_ecc(args=args, args_text_list=args_text_list)
    else:
        dnn_log_helper.log_and_crash(fatal_string=f"Code type {args.code_type} not implemented")


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
