import inspect
import itertools
import random
import string
import traceback
import torch
import time
import json

class FunctionAnalyzer:
    def __init__(self, func):
        self.func = func
        self.fail_type_combos = set()
        self.fail_arg_lengths = set()
        self.known_keys = set()
        self.success_cases = []
        self.return_types = {}
        self.error_stats = {"type_errors": 0, "arg_count_errors": 0, "other_errors": 0}
        self.total_attempts = 0
        self.exact_length = -1
        self.zero_confirmed = False
        self.zero_testing = False
        self.never_zero = False
        self.locked_types = {}  # key -> 'int', 'float', etc.
        self.type_priority = ['int', 'float', 'tensor', 'bool', 'str', 'none', 'list', 'dict']  # Order to try
        self.type_level = 1  # Start at 1, only allow first N types per key
        self.key_type_failures = {}  # key -> count

        self.max_params = 10
        sig = inspect.signature(func)
        self.kwargs_only = (
            len(sig.parameters) == 1 and 
            next(iter(sig.parameters.values())).kind == inspect.Parameter.VAR_KEYWORD
        )
        #print(f"[INIT] FunctionAnalyzer created for {func.__name__}")
        #print(f"[INIT] Initial state: fail_type_combos={self.fail_type_combos}, fail_arg_lengths={self.fail_arg_lengths}, known_keys={self.known_keys}")

    # --- type generators
    def random_primitive(self):
        t = random.choice(['int', 'float', 'bool', 'str', 'none'])
        #print(f"[random_primitive] Chosen type: {t}")
        if t == 'int': val = random.randint(-10000,10000)
        elif t == 'float': val = random.uniform(-10000,10000)
        elif t == 'bool': val = random.choice([True, False])
        elif t == 'str': val = ''.join(random.choices(string.ascii_letters+string.digits,k=5))
        else: val = None
        print(f"[random_primitive] Value: {val}")
        return val

    def random_tensor(self):
        shape = tuple(random.randint(1,3) for _ in range(random.randint(1,2)))
        #print(f"[random_tensor] Shape: {shape}")
        tensor = torch.randn(shape)
        #print(f"[random_tensor] Tensor: {tensor}")
        return tensor

    def random_list(self):
        l = [self.random_primitive() for _ in range(random.randint(1,5))]
        #print(f"[random_list] List: {l}")
        return l

    def random_dict(self):
        if self.known_keys:
            
            d = { k : self.random_primitive() for k in self.known_keys }
        else:
            d = { ''.join(random.choices(string.ascii_lowercase,k=3)) : self.random_primitive() for _ in range(random.randint(1,3)) }
        #print(f"[random_dict] Dict: {d}")
        return d

    def value_for_type(self, typ):
        #print(f"[value_for_type] Requested type: {typ}")
        if typ == 'int': val = random.randint(-1,1)
        elif typ == 'float': val = random.uniform(-1,1)
        elif typ == 'bool': val = random.choice([True, False])
        elif typ == 'str': val = ''.join(random.choices(string.ascii_letters+string.digits,k=5))
        elif typ == 'none': val = None
        elif typ == 'tensor': val = self.random_tensor()
        elif typ == 'list': val = self.random_list()
        elif typ == 'dict': val = self.random_dict()
        else: val = None
        #print(f"[value_for_type] Value: {val}")
        return val

    # --- main runners
    def run_permutations(self, max_params=3):
        #print(f"[run_permutations] Starting permutations with max_params={max_params}")
        TYPES = ['int', 'float', 'bool', 'str', 'none', 'tensor', 'list', 'dict']
        for n in range(1, max_params+1):
            #print(f"[run_permutations] Permuting argument count: {n}")
            all_combos = itertools.product(TYPES, repeat=n)
            for type_combo in all_combos:
                #print(f"[run_permutations] Trying type_combo: {type_combo}")
                values = [self.value_for_type(t) for t in type_combo]
                #print(f"[run_permutations] Values: {values}")
                self._try_call(type_combo, values)
                #print(f"[run_permutations] State after call: fail_type_combos={self.fail_type_combos}, fail_arg_lengths={self.fail_arg_lengths}, known_keys={self.known_keys}")

    def run_random(self, tries=1000):
        #print(f"[run_random] Starting random runs: tries={tries}, max_params={max_params}")
        TYPES = ['int', 'float', 'bool', 'str', 'none', 'tensor', 'list', 'dict']
        attempts = 0
        while attempts < tries:
            #print(f"[run_random] Attempt {attempts+1}/{tries}")
            #print(f"[run_random] Current state: exact_length={self.exact_length}, zero_confirmed={self.zero_confirmed}, zero_testing={self.zero_testing}, never_zero={self.never_zero}, fail_arg_lengths={self.fail_arg_lengths}")
            if self.exact_length >= 0:
                if self.zero_confirmed and self.exact_length == 0:
                    n = 0
                elif not self.zero_confirmed and self.exact_length == 0:
                    n = 1
                    self.zero_testing = True
                else:
                    n = self.exact_length
            else:
                if self.fail_arg_lengths:
                    remaining_choices = [i for i in range(0 if not self.never_zero else 1, self.max_params + 1) if i not in self.fail_arg_lengths]
                    #print(f"[run_random] Remaining choices for arg count: {remaining_choices}")
                    if not remaining_choices:
                        #print("[run_random] No remaining choices, breaking.")
                        break
                    n = random.choice(remaining_choices)
                else:
                    n = random.randint(0, self.max_params)
            #print(f"[run_random] Chosen argument count: {n}")
            if not self.kwargs_only:
                type_combo = tuple(random.choices(TYPES, k=n))
            else:
                # Allow repeated choices if not enough TYPES for known_keys
                if len(self.known_keys) == 0:
                    type_combo = tuple()
                else:
                    type_combo = tuple(random.choices(TYPES, k=len(self.known_keys)))
            print(f"[run_random] Chosen type_combo: {type_combo}")

            types_for_keys = []
            for k in self.known_keys:
                # Lock to explicit type if possible
                if k not in self.locked_types:
                    locked = self.infer_type_from_key(k)
                    if locked:
                        self.locked_types[k] = locked
                t = self.locked_types.get(k)
                if t:
                    types_for_keys.append(t)
                else:
                    # Use only the allowed types
                    types_for_keys.append(self.type_priority[0])
            values = [self.value_for_type(t) for t in types_for_keys]

            #print(f"[run_random] Values: {values}")
            self._try_call(type_combo, values)
            #print(f"[run_random] State after call: fail_type_combos={self.fail_type_combos}, fail_arg_lengths={self.fail_arg_lengths}, known_keys={self.known_keys}, error_stats={self.error_stats}")
            attempts += 1
            if len(self.fail_arg_lengths) >= self.max_params:
                print("[run_random] All possible arg counts known to fail, breaking early.")
                break
    def infer_type_from_key(self, key):
        key_lower = key.lower()
        if "float" in key_lower:
            return 'float'
        if "int" in key_lower:
            return 'int'
        if "tensor" in key_lower:
            return 'tensor'
        if "bool" in key_lower or key_lower.startswith('is_') or key_lower.startswith('has_'):
            return 'bool'
        if key_lower.endswith("idx") or key_lower.startswith("num") or key_lower in ('n', 'k', 'i', 'j', 'l', 'm'):
            return 'int'
        return None  # Means unknown

    def _try_call(self, type_combo, values):
        #print(f"[try_call] Attempting call with type_combo={type_combo}, values={values}")
        self.total_attempts +=1
        #print(f"[try_call] total_attempts={self.total_attempts}")
        start = time.perf_counter_ns()
        try:
            if self.kwargs_only:
                # Build keyword args with either discovered keys or random
                keys = list(self.known_keys) or [f"arg{i}" for i in range(len(values))]
                # If not enough keys, pad with random
                while len(keys) < len(values):
                    keys.append(''.join(random.choices(string.ascii_lowercase, k=5)))
                kw = {k: v for k, v in zip(keys, values)}
                print(kw)
                
                result = self.func(**kw)
            else:
        
                #print(f"[try_call] Calling function: {self.func.__name__} with values: {values}")
                result = self.func(*values) if len(type_combo) > 1 else (self.func() if len(type_combo) == 0 else self.func(values))
            duration = time.perf_counter_ns() - start
            rtype = type(result).__name__
            #print(f"[try_call] Call succeeded. Duration: {duration} ns, Return type: {rtype}, Result: {result}")
            try:
                serialized = json.dumps(result, default=str)
            except Exception:
                serialized = str(result)
            self.success_cases.append({
                "args": type_combo,
                "values": [str(v) for v in values],
                "duration_ns": duration,
                "return_type": rtype,
                "return_value": serialized
            })
            self.return_types.setdefault(rtype,0)
            self.return_types[rtype]+=1
            #print(f"[try_call] Updated return_types: {self.return_types}")
            if self.zero_testing and len(values) == 1:
                #print(f"[try_call] Zero testing revealed unreported arguments in function: {self.func.__name__} with types {type_combo} and values {values}")
                self.never_zero = True
                self.exact_length = 1
                self.zero_confirmed = False
                self.fail_arg_lengths.add(0)
                self.zero_testing = False
        except TypeError as e:
            msg = str(e)
            print(f"[try_call] TypeError: {msg} for types {type_combo} with values {values}")
            print(f"[try_call] State before handling TypeError: fail_arg_lengths={self.fail_arg_lengths}, zero_testing={self.zero_testing}, zero_confirmed={self.zero_confirmed}, never_zero={self.never_zero}, exact_length={self.exact_length}")
            if "too many positional arguments" in msg:
                self.fail_arg_lengths.add(len(values))
                self.error_stats["arg_count_errors"] +=1
                #print(f"[try_call] Added to fail_arg_lengths: {len(values)}")
            elif "takes " in msg and " positional arguments but " in msg and ("were given" in msg or "was given" in msg):
                end = "were given" if "were" in msg else "was given"
                reported_ideal = int(msg.split("takes ")[1].split(" positional arguments")[0])
                #print(f"[try_call] Reported ideal arg count: {reported_ideal}")
                if self.zero_testing and len(values) == 1:
                    self.zero_confirmed = True
                    self.exact_length = 0
                    self.zero_testing = False
                    print(f"[try_call] Zero confirmed, exact_length set to 0")
                
                else:
                    self.exact_length = reported_ideal
                    print(f"[try_call] exact_length set to {reported_ideal}")
                self.fail_arg_lengths.add(int(msg.split("but ")[1].split(end)[0]))
                self.error_stats["arg_count_errors"] +=1
                print(f"[try_call] Added to fail_arg_lengths: {int(msg.split('but ')[1].split(end)[0])}")
            else:
                self.fail_type_combos.add(type_combo)
                self.error_stats["type_errors"] +=1
                print(f"[try_call] Added to fail_type_combos: {type_combo}")
        except KeyError as e:
            # e.args[0] is the missing key
            print(f"[try_call] KeyError: {e.args[0]} for types {type_combo} with values {values}")
            print(f"[try_call] Context: known_keys={self.known_keys}, func={self.func.__name__}")
            # Optionally print all keyword arguments tried
            if self.kwargs_only:
                print(f"[try_call] Attempted keyword args: {kw}")
            # Optionally print stack trace
            traceback.print_exc()
            self.known_keys.add(e.args[0])
            if self.max_params < len(self.known_keys):
                self.max_params = len(self.known_keys)
            self.error_stats["key_errors"] = self.error_stats.get("key_errors", 0) + 1
        except Exception as e:
            msg = str(e)
            print(f"[try_call] Unexpected error: {msg} for types {type_combo} with values {values}")
            print(f"[try_call] State before handling Exception: known_keys={self.known_keys}, error_stats={self.error_stats}")
    
            self.error_stats["other_errors"] +=1
            #print(f"[try_call] Incremented other_errors")
            if self.zero_testing and len(values) == 1:
                #print(f"[try_call] Unspecified error while zero testing")
                self.zero_testing = False
                self.never_zero = False
                self.exact_length = -1
                self.zero_confirmed = False
        self.zero_testing = False  # Reset zero testing after each call
        #print(f"[try_call] End of call. State: fail_type_combos={self.fail_type_combos}, fail_arg_lengths={self.fail_arg_lengths}, known_keys={self.known_keys}, error_stats={self.error_stats}, exact_length={self.exact_length}, zero_confirmed={self.zero_confirmed}, zero_testing={self.zero_testing}, never_zero={self.never_zero}")

    # --- reporting
    def get_summary(self):
        print(f"[get_summary] Returning summary")
        print(f"[get_summary] total_attempts={self.total_attempts}, success_count={len(self.success_cases)}, fail_type_combos_count={len(self.fail_type_combos)}, fail_arg_lengths_count={len(self.fail_arg_lengths)}, return_types={self.return_types}, error_stats={self.error_stats}")
        return {
            "total_attempts": self.total_attempts,
            "success_count": len(self.success_cases),
            "fail_type_combos_count": len(self.fail_type_combos),
            "fail_arg_lengths_count": len(self.fail_arg_lengths),
            "return_types": self.return_types,
            "error_stats": self.error_stats
        }

    def dump_success_cases(self, path):
        print(f"[dump_success_cases] Dumping success cases to {path}")
        with open(path, 'w') as f:
            json.dump(self.success_cases, f, indent=2)
        print(f"[dump_success_cases] Dump complete")
