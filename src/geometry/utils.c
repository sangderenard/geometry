#include "geometry/utils.h"
#include "geometry/guardian_platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "geometry/graph_ops_handler.h"
#include "geometry/types.h"

// === Node Data Structures ===

// All type definitions are provided in the public header.


// --- DAG Manifest Structures Implementation ---
#include <stdlib.h>
#include <string.h>

Dag* dag_create(void) {
    Dag* dag = (Dag*)calloc(1, sizeof(Dag));
    return dag;
}

void dag_destroy(Dag* dag) {
    if (!dag) return;
    for (size_t i = 0; i < dag->num_manifests; ++i) {
        DagManifest* manifest = &dag->manifests[i];
        for (size_t l = 0; l < manifest->num_levels; ++l) {
            DagManifestLevel* level = &manifest->levels[l];
            for (size_t m = 0; m < level->num_mappings; ++m) {
                DagManifestMapping* mapping = &level->mappings[m];
                free(mapping->inputs);
                free(mapping->outputs);
            }
            free(level->mappings);
        }
        free(manifest->levels);
    }
    free(dag->manifests);
    free(dag);
}
void dag_add_manifest(Dag* dag, DagManifest* manifest) {
    if (!dag || !manifest) return;
    if (dag->num_manifests == dag->cap_manifests) {
        size_t new_cap = dag->cap_manifests ? dag->cap_manifests * 2 : 4;
        DagManifest* tmp = (DagManifest*)realloc(dag->manifests, new_cap * sizeof(DagManifest));
        if (!tmp) return;
        dag->manifests = tmp;
        dag->cap_manifests = new_cap;
    }
    dag->manifests[dag->num_manifests++] = *manifest;
}

size_t dag_num_manifests(const Dag* dag) {
    return dag ? dag->num_manifests : 0;
}

DagManifest* dag_get_manifest(const Dag* dag, size_t idx) {
    if (!dag || idx >= dag->num_manifests) return NULL;
    return &dag->manifests[idx];
}

size_t dag_manifest_num_levels(const DagManifest* manifest) {
    return manifest ? manifest->num_levels : 0;
}

DagManifestLevel* dag_manifest_get_level(const DagManifest* manifest, size_t level_idx) {
    if (!manifest || level_idx >= manifest->num_levels) return NULL;
    return &manifest->levels[level_idx];
}

size_t dag_level_num_mappings(const DagManifestLevel* level) {
    return level ? level->num_mappings : 0;
}

DagManifestMapping* dag_level_get_mapping(const DagManifestLevel* level) {
    return level ? level->mappings : NULL;
}

void dag_gather(const DagManifestMapping* mapping, void* out) {
    (void)mapping;
    (void)out;
    /* TODO: gather data from inputs */
}

void dag_scatter(const DagManifestMapping* mapping, void* data) {
    (void)mapping;
    (void)data;
    /* TODO: scatter data to outputs */
}

// --- NeuralNetwork implementation ---
#include <string.h>

NeuralNetwork* neuralnetwork_create(void) {
    NeuralNetwork* nn = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    return nn;
}

void neuralnetwork_destroy(NeuralNetwork* nn) {
    if (!nn) return;
    for (size_t d = 0; d < nn->num_dags; ++d) {
        dag_destroy(nn->dags[d]);
        for (size_t s = 0; s < nn->num_steps[d]; ++s) {
            free(nn->steps[d][s]);
        }
    }
    free(nn);
}

void neuralnetwork_register_function(NeuralNetwork* nn, const char* name, NNForwardFn forward, NNBackwardFn backward) {
    if (!nn || !name) return;
    if (nn->function_repo.num_entries >= NN_MAX_FUNCTIONS) return;
    NeuralNetworkFunctionEntry* e = &nn->function_repo.entries[nn->function_repo.num_entries++];
    e->name = name;
    e->forward = forward;
    e->backward = backward;
}

void neuralnetwork_set_step_function(NeuralNetwork* nn, size_t dag_idx, size_t step_idx, const char* function_name, void* user_data) {
    if (!nn || dag_idx >= nn->num_dags || step_idx >= NN_MAX_STEPS) return;
    NeuralNetworkStep* step = nn->steps[dag_idx][step_idx];
    if (!step) {
        step = (NeuralNetworkStep*)calloc(1, sizeof(NeuralNetworkStep));
        nn->steps[dag_idx][step_idx] = step;
        if (step_idx >= nn->num_steps[dag_idx]) nn->num_steps[dag_idx] = step_idx + 1;
    }
    for (size_t i = 0; i < nn->function_repo.num_entries; ++i) {
        NeuralNetworkFunctionEntry* e = &nn->function_repo.entries[i];
        if (strcmp(e->name, function_name) == 0) {
            step->forward = e->forward;
            step->backward = e->backward;
            step->user_data = user_data;
            break;
        }
    }
}

void neuralnetwork_forward(NeuralNetwork* nn) {
    (void)nn; // TODO
}

void neuralnetwork_backward(NeuralNetwork* nn) {
    (void)nn; // TODO
}

void neuralnetwork_forwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx) {
    (void)nn; (void)dag_idx; (void)step_idx; // TODO
}

void neuralnetwork_backwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx) {
    (void)nn; (void)dag_idx; (void)step_idx; // TODO
}

// === Consolidated Guardian Implementation ===

// === GuardianNode structure ===
// Full structure:
// typedef struct Node {
//     char* id;
//     GuardianStencilSet stencil_set; // Stencil set for the node
//     GuardianObjectSet object_set; // Object set for the node listing the thread lock token, guardian pointer, and the object pointer token
//     GuardianObjectSet feature_set; // Set of features associated with the node as an arbitrary guardian object
//     GuardianSet compatible_edges; // Set of compatible edge_type identities for at least one point on one stencil this node
//     GuardianList internal_edges; // List of internal edges (edges that are not exposed to the outside world, but are used internally for node processing
//     struct Emergence* emergence = NULL;
// } Node;
size_t guardian_sizeof(NodeFeatureType type) {
    switch (type) {
        case NODE_FEATURE_TYPE_FLOAT:
            return sizeof(float);
        case NODE_FEATURE_TYPE_DOUBLE:
            return sizeof(double);
        case NODE_FEATURE_TYPE_INT:
            return sizeof(int);
        //case NODE_FEATURE_TYPE_UINT:
        //    return sizeof(unsigned int);
        case NODE_FEATURE_TYPE_STRING:
            return sizeof(char*);
        case NODE_FEATURE_TYPE_BOOLEAN:
            return sizeof(boolean);
        case NODE_FEATURE_IDX_HEAP:
            return sizeof(GuardianHeap);
        case NODE_FEATURE_IDX_GUARDIAN:
            return sizeof(TokenGuardian);
        case NODE_FEATURE_IDX_OBJECT_SET:
            return sizeof(GuardianObjectSet);
        case NODE_FEATURE_IDX_POINTER_TOKEN:
            return sizeof(GuardianPointerToken);
        case NODE_FEATURE_IDX_LIST:
            return sizeof(GuardianList);
        case NODE_FEATURE_IDX_LINKED_LIST:
            return sizeof(GuardianLinkedList);

        default:
            return 0; // Unsupported type
    }
}
TokenGuardian * find_token_authority(TokenGuardian *g) {
    if (!g) return NULL;
    if (g->token_host == g) return g; // Found the root guardian
    return find_token_authority(g->token_host); // Recursively find the root guardian
}//	GuardianParallelList thread_tokens_locks_buffers_threads; // the manager database for threads

GuardianGeneology * guardian_create_geneology(TokenGuardian* g, GuardianStencilSet * stencil_set, NodeOrientationNature * orientation, ParametricDomain * domain) {
    if (!g) return NULL;
    GuardianGeneology* geneology = (GuardianGeneology*)instantiate_on_input_cache(NODE_FEATURE_IDX_GENEALOGY);
    if (!geneology) return NULL;
    geneology->stencil_set = stencil_set;
    geneology->domain = domain;
    geneology->self->guardian = g; // Link the geneology to the guardian
    return geneology;
}

TokenGuardian * guardian_initialize(TokenGuardian* parent, size_t num_threads) {
    TokenGuardian * g = guardian_global_cache_create(NODE_FEATURE_IDX_GUARDIAN);
    GuardianObjectSet* self = guardian_initialize_obj_set(g);
    if (!parent) {
		GuardianToken* pointer_token = instantiate_on_input_cache(NODE_FEATURE_IDX_POINTER_TOKEN);
        self->guardian_pointer_token = pointer_token; // Default token for the guardian
        g->main_lock = guardian_create_token_lock(g, self->guardian_pointer_token);
		g->token_host = g; // Self-referential token host
	}
	else {
        g->token_host = parent;
		self->guardian_pointer_token = guardian_create_pointer_token(parent, g, NODE_FEATURE_IDX_GUARDIAN );
		g->main_lock = guardian_create_token_lock(parent, self->guardian_pointer_token);
	}
    


    g->allocation_growth_factor = 1.5; // Default growth factor
    g->concurrent_threads = num_threads > 0 ? num_threads : 4; // Default to 4 threads if not specified
    g->max_allocation = 1024 * 1024 * 1024; // Default max allocation size (1 GB)
    g->min_allocation = 1024; // Default min allocation size (1 KB)

    g->data = instantiate_on_input_cache(NODE_FEATURE_IDX_HEAP); // Initialize data structure

    g->max_threads = g->min_allocation;
    g->using_buffered_io = false; // Default to not using buffered I/O
    g->auto_contiguous_allocation = true; // Default to automatic contiguous allocation
    g->auto_synchronization = true; // Default to automatic synchronization
    g->max_stack_count = g->max_threads;
    g->max_stack_size = 1024 * 1024; // Default stack size (1 MB)
    g->default_feature_type = NODE_FEATURE_TYPE_FLOAT; // Default feature type for nodes
    g->default_stencil_set = guardian_create_stencil_set(g, guardian_create_list(g, guardian_create_stencil(g, RECT_STENCIL_DEFAULT), NODE_FEATURE_IDX_STENCIL, NULL)); // Default stencil set for nodes
    g->default_node_orientation = NODE_ORIENTATION_DOMAIN_PARALLEL; // Default node orientation
    g->default_parametric_domain = parametric_domain_create(3); // Default 3D parametric domain
    g->default_geneology = guardian_create_geneology(g, g->default_stencil_set, g->default_node_orientation, g->default_parametric_domain);
    g->next_token = id_dispenser();
}

GuardianToken * guardian_get_new_token(TokenGuardian* g, int size){
    GuardianToken * return_token = instantiate_in_cache(NODE_FEATURE_IDX_TOKEN);
    if (!size || size <= 0) {
        size = 1; // Default size if not specified
    }
    if (!g) {
        return_token->token = GUARDIAN_NOT_USED;
    } else {
        return_token = guardian_register_in_pool(g, return_token, size);
        return_token->token = id_dispenser();
    }
    return return_token;
}
// Generate a token for a pointer without exposing the pointer itself
GuardianToken * guardian_create_pointer_token(TokenGuardian* g, void* ptr, NodeFeatureType type) {
    if (!g) {
        TokenGuardian dummy = {GUARDIAN_NOT_USED};
        g = &dummy; // Use a dummy guardian if none is provided
    }
    GuardianToken * dummy_data = guardian_create_dummy();;
    if (!ptr) return dummy_data; // Return an empty token if the pointer is null
    GuardianToken * token;
    token->token = guardian_register_in_pool(g, guardian_get_unregistered_stack(g, type));

    if (token->token == 0) {
        // Handle error: unable to create pointer token
        GuardianToken * pointer_token = guardian_get_new_token(g, guardian_sizeof(type));
        boolean success = guardian_register_out_of_heap_pointer_token(g, pointer_token, ptr);
        if (!success) {
            // Handle error: unable to register pointer token
            return NULL;
        }
        return pointer_token;
    }
    return token;
}

GuardianToken * guardian_create_lock_token(TokenGuardian* g) {
    if (!g) {
        TokenGuardian * dummy = guardian_create_dummy();
        g = dummy; // Use a dummy guardian if none is provided
    }
    GuardianToken * lock_token = guardian_get_new_token(g, sizeof(mutex_t));
    if (lock_token->token == 0) return NULL;
    mutex_t* mutex = instantiate_on_input_cache(NODE_FEATURE_IDX_MUTEX_T);
    if (!mutex) return NULL;
    guardian_mutex_init(mutex);
    // Register the mutex in the guardian's lock pool
    boolean success = guardian_register_out_of_heap_pointer_token(g, lock_token, mutex);
    if (!success) {
        free(mutex);
        return NULL;
    }
    return lock_token;
}

GuardianToken * guardian_create_dummy() {
    TokenGuardian * dummy_guardian = instantiate_on_input_cache(NODE_FEATURE_IDX_GUARDIAN)    ;

    dummy_guardian->is_locked = false;
    dummy_guardian->is_initialized = false;
    dummy_guardian->is_contiguous = false;
    dummy_guardian->is_synchronized = false;

    dummy_guardian->self = guardian_initialize_obj_set(&dummy_guardian);
    dummy_guardian->main_lock = guardian_create_token_lock(&dummy_guardian, dummy_guardian->self->guardian_pointer_token);

    dummy_guardian->max_threads = 1;
    dummy_guardian->min_allocation = 0;
    dummy_guardian->max_allocation = 0;
    dummy_guardian->allocation_growth_factor = 1.0;
    dummy_guardian->concurrent_threads = 1;

    dummy_guardian->using_buffered_io = false;
    dummy_guardian->auto_contiguous_allocation = false;
    dummy_guardian->auto_synchronization = false;

    dummy_guardian->heap = NULL;
    dummy_guardian->data = NULL;
    dummy_guardian->stack_memory_map = &(GuardianMap){0};

    dummy_guardian->self->guardian_pointer_token->token = GUARDIAN_NOT_USED;
    dummy_guardian->next_token->token = GUARDIAN_NOT_USED;

    return true; // Success
}

GuardianLinkedList * guardian_linked_list_set_chain(GuardianLinkedList * list, GuardianLinkNode ** chain, int type, int chain_length) {
    if (!list || !chain) return NULL;
    list->left = chain;
    list->right = chain + (chain_length - 1) * sizeof(GuardianLinkNode*);
    list->feature_type = type;
    list->size = chain_length;
    list->max_size = MAX_LINK_LIST_SIZE; // Set the maximum size for the linked
    return list;
}

GuardianList * guardian_parallel_list_set_list(int index, GuardianParallelList * list, GuardianList * data) {
    int num_lists = list ? list->num_lists : 0;
    if (!list || !data || index < 0 || index >= list->num_lists) return false;
    if (index >= num_lists) {
        graph_ops_list.set(list->lists, index, data); // Ensure the list has enough space
        list->num_lists++;
    } else if (index > 0 && index < num_lists) {
        // Ensure the list has enough space
        graph_ops_list.set(list->lists, index, data);
    }
    return true;
}

const MAX_LINK_LIST_SIZE = 1000000; // Maximum size for linked lists
GuardianLinkedList * guardian_create_linked_list(TokenGuardian* g, int initialized_length, int type, void** data) {
    GuardianLinkedList * list = instantiate_on_input_cache(NODE_FEATURE_IDX_LINKED_LIST);
    GuardianLinkNode ** node = instantiate_chain_on_input_cache(type, initialized_length);
    list = guardian_linked_list_set_chain(list, node, type, initialized_length);
    boolean success = guardian_linked_list_straight_write(list, data, initialized_length);
    if (!success) {
        // Handle error: unable to write to linked list
        return NULL;
    }
    return list;
}

GuardianList* guardian_create_list(TokenGuardian* g, int initialized_length, int type, void** data) {
    GuardianList* list = instantiate_on_input_cache(NODE_FEATURE_IDX_LIST);
    list->pointer_to_index = guardian_create_linked_list(g, initialized_length, type, data);
    list = guardian_refresh_from_pointer_to_index(list);
    return list;
}

GuardianStencilSet * guardian_create_stencil_set_internal_(TokenGuardian* g, GuardianList* stencils, GuardianList* orthagonalities, GuardianList* orientations) {
    GuardianStencilSet * set = instantiate_on_input_cache(NODE_FEATURE_IDX_STENCIL_SET);
    GuardianObjectSet * self = guardian_initialize_obj_set(g);
    set->stencils_orthagonalities_orientations = guardian_create_parallel_list(self->guardian, 3);
    
	boolean success = guardian_parallel_list_set_list(0, set->stencils_orthagonalities_orientations, stencils);
    if (!success) {
        // Handle error: unable to set stencils
		return NULL;
    }
    success = guardian_parallel_list_set_list(1, set->stencils_orthagonalities_orientations, orthagonalities);
    if (!success) {
        // Handle error: unable to set orthagonalities
        return NULL;
	}
    success = guardian_parallel_list_set_list(2, set->stencils_orthagonalities_orientations, orientations);
    if (!success) {
        // Handle error: unable to set orientations
		return NULL;
    }
    return set;
}

boolean guardian_token_lock_checkout_check(TokenGuardian* g, GuardianToken* token, boolean reentrant) {
    if (!g || !token) return false;
    
    GuardianToken* guardian_object_token = g->self->guardian_pointer_token;
    if (!guardian_object_token || guardian_object_token->token == 0) {
        // Handle error: invalid guardian object token
        return false;
    } else if (reentrant && guardian_object_token->token == token->token) {
        // Token is already owned by this thread, can skip locking
        return true;
    }
    if (!memory_ops_locks_checked_out(token->token)) {
        // Token is not in use, can be locked
        return true;
    }
    // Token is already in use, cannot lock
    return false;
}

boolean guardian_lock_with_timeout(TokenGuardian* g, GuardianToken* guardian_lock_token, int duration, boolean reentrant) {
    boolean set_lock = false;
    if (!duration) {
        duration = 1000; // Default to 1 second if no duration is specified
	}   
    if (reentrant) {
        if (memory_ops_lock_checkout_check(g, guardian_lock_token->token, reentrant)) {
            // If the token is already owned by this thread, we can skip locking
            set_lock = true;
        }
    }

    while (!set_lock && memory_ops_try_lock(g, guardian_lock_token->token) == 0 && duration > 0 ) {
        // Wait for a short period before retrying
        usleep(1000); // Sleep for 1 millisecond
		duration -= 1; // Decrease the remaining duration
	}

    if (duration <= 0) {
        // Timeout occurred
        return false;
    }
    // Lock acquired successfully
	set_lock = true;
    if (set_lock) {
        
        return memory_ops_set_lock(guardian_lock_token, g);
    }
    return false; // Unable to acquire lock
}

GuardianToken* guardian_register_in_pool(TokenGuardian* g, GuardianStack* unregistered_stack) {
    if (!g) {
        g = guardian_create_dummy(); // Create a dummy guardian if none is provided
    }
    GuardianToken * object_token = g->heap->self->guardian_lock_token;
    guardian_lock_with_timeout(g, object_token , 1000, false);
	boolean success = guardian_add_stack_to_pool(g, unregistered_stack);
	
    if (!success) {
        // Handle error: unable to register stack in pool
        memory_ops_unlock(object_token);
        return NULL;
    }
    g->is_contiguous = false; // Reset contiguous flag for the stack
	void* stack_location = unregistered_stack->data; // Get the stack location
    if (g->auto_contiguous_allocation) {
        // Attempt to allocate contiguous memory for the stack
        if (!guardian_try_contiguous_wait(g)) {
            // Handle error: unable to allocate contiguous memory
            memory_ops_unlock(object_token);
            return NULL;
        }

	}
    guardian_try_contiguous_nowait(g);
    
	GuardianToken * memory_token = guardian_get_new_token(g, sizeof(GuardianStack));
    if (memory_token->token == 0) {
        // Handle error: unable to get a new token
        memory_ops_unlock(object_token);
        return NULL;
    }
    
    if (memory_token->token == 0) {
        // Handle error: unable to create pointer token for the stack
        memory_ops_unlock(object_token);
        return NULL;
    }
	// Successfully registered the stack and created a new token
	memory_ops_unlock(object_token);
    return memory_token;
}

boolean guardian_ask_for_block(TokenGuardian* g, GuardianObjectSet* obj_set, int count) {
    if (!g || !obj_set || count <= 0) return false;
    // Check if the guardian is initialized
    if (!g->is_initialized) {
        // Handle error: guardian is not initialized
        return false;
    }
    // Check if the object set is valid
    if (obj_set->guardian_pointer_token->token == 0) {
        // Handle error: invalid object set
        return false;
    }
    boolean contiguous_success = graph_ops_heap.math_ops.make_contiguous_no_wait(g->heap);
    boolean gap_success = graph_ops_heap.math_ops.gap_inventory(g->heap);
    if(!gap_success){
        contiguous_success = graph_ops_heap.math_ops.make_contiguous_wait_timeout(g->heap);
        if (!contiguous_success) {
            if (g->auto_contiguous_allocation) {
                // Attempt to allocate contiguous memory block
                contiguous_success = graph_ops_heap.math_ops.make_contiguous_force(g->heap);
                if (!contiguous_success) {
                    // Handle error: unable to allocate contiguous memory block
                    return false;
                } else {
                    // Successfully allocated the block
                    return true;
                }
            }
            
            return false;
        }
    }
    return true; // Successfully allocated the block
}

GuardianObjectSet * guardian_create_node_object_set(TokenGuardian* g, int count) {
	GuardianObjectSet * obj_set = guardian_initialize_obj_set(g);
    boolean lock_state = memory_ops_unlock(obj_set->guardian, obj_set->guardian_lock_token->token);
    if (!lock_state) {
        // Handle error: unable to unlock the guardian lock token
        return NULL;
	}

    if (obj_set->guardian_pointer_token->token == 0) {
        // Handle error: unable to create pointer token
		return NULL;
    }
    GuardianToken * authorized_tracked_token =
        guardian_register_in_pool(g, guardian_ask_for_block(g, obj_set->guardian_pointer_token, count));
    obj_set->guardian_pointer_token = authorized_tracked_token;
    return obj_set;
    
}
    
GuardianObjectSet * guardian_initialize_obj_set( TokenGuardian* g){
    GuardianObjectSet* obj_set = NULL;
    if (!g) {
	    obj_set = instantiate_on_input_cache(NODE_FEATURE_IDX_OBJECT_SET);
        g = guardian_create_dummy();
        obj_set->guardian = g; // Link the object set to the guardian
    }
    else {
        obj_set = instantiate_on_heap(NODE_FEATURE_IDX_OBJECT_SET);
        obj_set->guardian = g;
    }
    obj_set->guardian_lock_token = guardian_create_lock_token(g);
    return obj_set;
}


