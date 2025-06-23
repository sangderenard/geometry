// Node arithmetic operators (feature-wise)
inline Node* operator+(const Node& a, const Node& b) { return node_add(&a, &b); }
inline Node* operator-(const Node& a, const Node& b) { return node_sub(&a, &b); }
inline Node* operator*(const Node& a, const Node& b) { return node_mul(&a, &b); }
inline Node* operator/(const Node& a, const Node& b) { return node_div(&a, &b); }
inline Node* operator+(const Node& a, double s) { return node_add_scalar(&a, s); }
inline Node* operator*(const Node& a, double s) { return node_mul_scalar(&a, s); }
inline Node* operator+(double s, const Node& a) { return node_add_scalar(&a, s); }
inline Node* operator*(double s, const Node& a) { return node_mul_scalar(&a, s); }

// Geneology set-theoretic operators
inline Geneology* operator|(const Geneology& a, const Geneology& b) { return geneology_union(&a, &b); }
inline Geneology* operator&(const Geneology& a, const Geneology& b) { return geneology_intersection(&a, &b); }
inline Geneology* operator-(const Geneology& a, const Geneology& b) { return geneology_difference(&a, &b); }
// ...add more as needed (e.g., ^ for symmetric difference)
