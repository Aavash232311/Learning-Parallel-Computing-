The Standard Softmax logic from the flash attention paper

$$S = QK^T, \quad P = \text{softmax}(S), \quad O = PV$$

Let's take $O = PV$

In topics like thermodynamics ( for different thermodynamics processes ) or let's say wave function we do partial differentiation but here $dO$ is the output of matrix multiplication between $P$ and value $V$.

$$dO = P \, dV + dP \, V$$

In neural networks we care about $L(O)$ because it tells us how to change the model's parameters to reduce the loss.

Let us consider a flow state:

$$\text{Input} \rightarrow \text{Layer A} \rightarrow \text{Layer B} \rightarrow \text{Loss } L$$

When we are trying to compute the gradient for Layer A, the gradient coming from B is called the upstream gradient.

For a scalar function $L(O)$

> **Note:** $:$ meaning inner product basically multiply each element and then sum.

$$dL = \sum_{i,j} (L_O)_{ij} \, dO_{ij}$$

$$dL = L_O : dO$$

The upstream gradient $G = L_O$

$$dL = G : (P \, dV + dP \, V)$$
$$dL = G : (P \, dV) + G : (dP \, V)$$

Lets write the above term in index notation again:

$$G : (P \, dV) = \sum_{i,j} G_{ij} (P \, dV)_{ij} \quad \text{--- i)}$$
$$G : (dP \, V) = \sum_{i,j} G_{ij} (dP \, V)_{ij} \quad \text{--- ii)}$$

Lets understand why that transpose came into place by proving one of the equations.

From equation i)

$$G : (P \, dV) = \sum_{i,j} G_{ij} (P \, dV)_{ij}$$

Let us only take the term $P \, dV$ ( Note:- this is not thermodynamics or waive function the output is defined as the result of something so we cannot keep something constant)

$$(P \, dV)_{ij} = \sum_{k} P_{ik} (dV)_{kj} \quad \text{--- iii)}$$

> **Note:** From the attention score formula $P$ which is the output of the softmax and then $V$ which is the value is matrix multiplied.

Now multiply that term $(P \, dV)_{ij}$ with upstream gradient $G$

From equation iii)

$P_{ik} (dV)_{kj}$ and $G_{ij}$ will have an inner product.

$$G : (P \, dV) = \sum_{i,j,k} G_{ij} P_{ik} (dV)_{kj}$$

We are now regrouping by multiplying with $(dV)_{kj}$

$$\sum_{k,j} \left( \sum_{i} P_{ik} G_{ij} \right) (dV)_{kj} \quad \text{--- iv)}$$

Basically you can group this with the involved terms.

Now by the definition of transpose from school level math.

$$(P^T)_{ki} = P_{ik}$$

Common point of confusing here value of the index are same:

$$\sum_{k,j} \left( \sum_{i} (P^T)_{ki} G_{ij} \right) (dV)_{kj} \quad \text{from eq iv)}$$

$$G : (P \, dV) = (P^T G) : dV$$

Reading off the gradient for the first and second part of the above equation.

$$dL = (P^T G) : dV$$

$$L_V = P^T L_O \quad \text{--- a)}$$

$$dL = (G V^T) : dP$$

$$L_P = L_O V^T \quad \text{--- b)}$$

Let's unfold exactly what is going on at the matrix level.

The Value matrix has shape $(3 \times 3)$:

$$
V = \begin{bmatrix}
v_{11} & v_{12} & v_{13} \\
v_{21} & v_{22} & v_{23} \\
v_{31} & v_{32} & v_{33}
\end{bmatrix}
$$

---

### 2. Attention Probability Matrix ($P$)

Each row of $P$ is the result of applying the `softmax` function across the corresponding row of the attention score matrix $S$.

Let the row-wise denominator sums be defined as:

- Row 1: $\sum e^{s_1} = e^{s_{11}} + e^{s_{12}} + e^{s_{13}}$
- Row 2: $\sum e^{s_2} = e^{s_{21}} + e^{s_{22}} + e^{s_{23}}$
- Row 3: $\sum e^{s_3} = e^{s_{31}} + e^{s_{32}} + e^{s_{33}}$

$$
P = \text{softmax}(S) = \begin{bmatrix}
\frac{e^{s_{11}}}{\sum e^{s_1}} & \frac{e^{s_{12}}}{\sum e^{s_1}} & \frac{e^{s_{13}}}{\sum e^{s_1}} \\
\frac{e^{s_{21}}}{\sum e^{s_2}} & \frac{e^{s_{22}}}{\sum e^{s_2}} & \frac{e^{s_{23}}}{\sum e^{s_2}} \\
\frac{e^{s_{31}}}{\sum e^{s_3}} & \frac{e^{s_{32}}}{\sum e^{s_3}} & \frac{e^{s_{33}}}{\sum e^{s_3}}
\end{bmatrix} = \begin{bmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33}
\end{bmatrix}
$$

---

### 3. Output Matrix ($O = PV$)

Multiplying rows of $P$ by columns of $V$ gives us the fully expanded elements of the output matrix $O$:

$$
O = \begin{bmatrix}
p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31} & p_{11}v_{12} + p_{12}v_{22} + p_{13}v_{32} & p_{11}v_{13} + p_{12}v_{23} + p_{13}v_{33} \\
p_{21}v_{11} + p_{22}v_{21} + p_{23}v_{31} & p_{21}v_{12} + p_{22}v_{22} + p_{23}v_{32} & p_{21}v_{13} + p_{22}v_{23} + p_{23}v_{33} \\
p_{31}v_{11} + p_{32}v_{21} + p_{33}v_{31} & p_{31}v_{12} + p_{32}v_{22} + p_{33}v_{32} & p_{31}v_{13} + p_{32}v_{23} + p_{33}v_{33}
\end{bmatrix}
$$

Which maps directly to the standard layout:

$$
O = \begin{bmatrix}
O_{11} & O_{12} & O_{13} \\
O_{21} & O_{22} & O_{23} \\
O_{31} & O_{32} & O_{33}
\end{bmatrix}
$$

The upstream gradient for \( P = \text{softmax}(S) = \frac{\partial L}{\partial P} \)

**Quotient Rule:**

The quotient rule for derivatives states:

$$
\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}
$$

where \(u\) and \(v\) are functions of \(x\), and \(u'\) and \(v'\) are their respective derivatives.

Now, what we want to do is evaluate this row-wise for the attention probability matrix $P$.

Let's first look closely at the numerator and the denominator of the softmax fraction.

For example, consider the position $p_{11}$:

- **Numerator ($N$):** $N = e^{s_{11}}$
- **Denominator ($D$):** $D = e^{s_{11}} + e^{s_{12}} + e^{s_{13}}$

Let's take the partial derivative with respect to the input score $s_{11}$:

- The derivative of the numerator with respect to $s_{11}$ is:
  $$N' = \frac{\partial N}{\partial s_{11}} = e^{s_{11}}$$

- The derivative of the denominator with respect to $s_{11}$ is:
  $$D' = \frac{\partial D}{\partial s_{11}} = e^{s_{11}}$$

The Full Chain Rule across the Row

Here we apply the chain rule to compute the gradient with respect to the input logit $s_{11}$.

Because the denominator is a shared sum across the entire row, changing $s_{11}$ alters the value of _every single probability element_ in that row ($p_{11}$, $p_{12}$, and $p_{13}$). This row-wise coupling is exactly why parallel reduction in a GPU kernel requires careful shared memory communication or warp shuffles.

Using the total derivative chain rule, we must sum the gradient contributions flowing back through all elements in that row:

In this case, we have $\frac{\partial L}{\partial P}$ as our **upstream gradient** and $\frac{\partial P}{\partial S_{11}}$ as our **local gradient**. Therefore,

$$
\frac{\partial L}{\partial S_{11}}
=
\frac{\partial L}{\partial P}
\cdot
\frac{\partial P}{\partial S_{11}}
$$

This follows directly from the chain rule.

More explicitly, if $P$ has multiple elements, we can write:

$$
\frac{\partial L}{\partial S_{11}}
=
\frac{\partial L}{\partial P_{11}}
\frac{\partial P_{11}}{\partial S_{11}}
+
\frac{\partial L}{\partial P_{12}}
\frac{\partial P_{12}}{\partial S_{11}}
+
\frac{\partial L}{\partial P_{13}}
\frac{\partial P_{13}}{\partial S_{11}}
$$

In a more compact notation using the upstream gradient components $\frac{\partial L}{\partial P_{1j}}$:

$$
\frac{\partial L}{\partial S_{11}}
=
\sum_{j=1}^{3}
\frac{\partial L}{\partial P_{1j}}
\frac{\partial P_{1j}}{\partial S_{11}}
$$

In a more compact notation using the upstream gradient components $(L_P)_{1j}$:

$$
\frac{\partial L}{\partial s_{11}} = \sum_{j=1}^{3} \frac{\partial L}{\partial p_{1j}} \frac{\partial p_{1j}}{\partial s_{11}}
$$

$$\frac{\partial L}{\partial s_{11}} = \left( \frac{\partial L}{\partial p_{11}} \cdot \frac{\partial p_{11}}{\partial s_{11}} \right) + \left( \frac{\partial L}{\partial p_{12}} \cdot \frac{\partial p_{12}}{\partial s_{11}} \right) + \left( \frac{\partial L}{\partial p_{13}} \cdot \frac{\partial p_{13}}{\partial s_{11}} \right) \quad \text{--- (Equation V)}$$

Now in our example, at the position $p_{11}$:

$$p_{11} = \frac{e^{s_{11}}}{e^{s_{11}} + e^{s_{12}} + e^{s_{13}}}$$

Now taking the partial derivative of $p_{11}$ with respect to $s_{11}$ using the quotient rule:

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{e^{s_{11}} \left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right) - e^{s_{11}} \left( e^{s_{11}} \right)}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2}$$

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{e^{s_{11}} e^{s_{11}} + e^{s_{11}} e^{s_{12}} + e^{s_{11}} e^{s_{13}} - e^{s_{11}} e^{s_{11}}}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2}$$

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{e^{s_{11}} e^{s_{12}} + e^{s_{11}} e^{s_{13}}}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2}$$

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{\left(e^{s_{11}} e^{s_{11}} + e^{s_{11}} e^{s_{12}} + e^{s_{11}} e^{s_{13}}\right) - e^{s_{11}} e^{s_{11}}}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2}$$

Now, factor out $e^{s_{11}}$ from the first group in the numerator:

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{e^{s_{11}} \left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right) - \left(e^{s_{11}}\right)^2}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2}$$

Next, split the fraction into two separate terms across the minus sign:

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{e^{s_{11}} \left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2} - \frac{\left(e^{s_{11}}\right)^2}{\left( e^{s_{11}} + e^{s_{12}} + e^{s_{13}} \right)^2}$$

Cancel out the shared terms in the first fraction:

$$\frac{\partial p_{11}}{\partial s_{11}} = \frac{e^{s_{11}}}{e^{s_{11}} + e^{s_{12}} + e^{s_{13}}} - \left( \frac{e^{s_{11}}}{e^{s_{11}} + e^{s_{12}} + e^{s_{13}}} \right)^2$$

Since $p_{11} = \frac{e^{s_{11}}}{e^{s_{11}} + e^{s_{12}} + e^{s_{13}}}$, we can substitute it directly back into the terms:

$$\frac{\partial p_{11}}{\partial s_{11}} = p_{11} - p_{11}^2$$

There's nothing to be scared of. Think of this as multiplying by the upstream gradient; it's just the quotient rule applied to the derivative term and then rearranging the terms.

For position $p_{12}$, because this is a partial derivative, everything else is a constant.

$$\frac{\partial p_{12}}{\partial s_{11}} = \frac{0 \cdot \left(e^{s_{11}} + e^{s_{12}} + e^{s_{13}}\right) - e^{s_{12}} e^{s_{11}}}{\left(e^{s_{11}} + e^{s_{12}} + e^{s_{13}}\right)^2}$$

$$\frac{\partial p_{12}}{\partial s_{11}} = \frac{- e^{s_{11}} e^{s_{12}}}{\left(e^{s_{11}} + e^{s_{12}} + e^{s_{13}}\right)^2}$$

$$= -p_{11} p_{12}$$

In a similar way for position $p_{13}$:

$$\frac{\partial p_{13}}{\partial s_{11}} = -p_{11} p_{13}$$

Remember we are doing a row wise opreation here.

From Equation V:

$$\frac{\partial L}{\partial s_{11}} = \frac{\partial L}{\partial p_{11}} (p_{11} - p_{11}^2) + \frac{\partial L}{\partial p_{12}} (-p_{11} p_{12}) + \frac{\partial L}{\partial p_{13}} (-p_{11} p_{13})$$

In a similar way we can calculate for row 2 and row 3.

For s₁₂, we take:

$$\frac{\partial L}{\partial s_{12}} = \left(\frac{\partial L}{\partial p_{11}} \cdot (-p_{11}p_{12})\right) + \left(\frac{\partial L}{\partial p_{12}} \cdot p_{12}(1-p_{12})\right) + \left(\frac{\partial L}{\partial p_{13}} \cdot (-p_{12}p_{13})\right)$$

For s₁₃, we take:

$$\frac{\partial L}{\partial s_{13}} = \left(\frac{\partial L}{\partial p_{11}} \cdot (-p_{11}p_{13})\right) + \left(\frac{\partial L}{\partial p_{12}} \cdot (-p_{12}p_{13})\right) + \left(\frac{\partial L}{\partial p_{13}} \cdot p_{13}(1-p_{13})\right)$$

Now lets take a look at the original matrix.

$$
P = \text{softmax}(S) =
\begin{bmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33}
\end{bmatrix}
$$

Lets pull out gradient vector from equation vi meaning from the first row for example

$$\frac{\partial L}{\partial s_{1:}} = \begin{bmatrix} \frac{\partial L}{\partial s_{11}} & \frac{\partial L}{\partial s_{12}} & \frac{\partial L}{\partial s_{13}} \end{bmatrix}$$

For P which is the output of the softmax

$$\frac{\partial L}{\partial p_{1:}} = \begin{bmatrix} \frac{\partial L}{\partial p_{11}} & \frac{\partial L}{\partial p_{12}} & \frac{\partial L}{\partial p_{13}} \end{bmatrix}$$

will we wrap those vectors like this, if you multiply then you get the same result as above and remember all this for a single row.

$$
\frac{\partial L}{\partial S_1} = \frac{\partial L}{\partial P_1} \cdot
\begin{bmatrix}
p_{11}(1-p_{11}) & -p_{11}p_{12} & -p_{11}p_{13} \\
-p_{11}p_{12} & p_{12}(1-p_{12}) & -p_{12}p_{13} \\
-p_{11}p_{13} & -p_{12}p_{13} & p_{13}(1-p_{13})
\end{bmatrix}
$$

Now we will split this into Jacobian matrix and if you have doubt then multiply and see a small example its the same.

$$J(P_1) = \text{diag}(P_1) - P_1^T P_1$$

By the definition of a diagonal matrix, expanding the above equation, we get:

$$
J_{ij}
=
\delta_{ij} p_{1i}
-
p_{1j}^{T} P
-
\text{function } J
$$

Let $dY$ be the upstream gradient.

Basically, we multiply by the incoming gradient. I like to call it $G$, but let it be $dY$ for now.

$$
dX_i
=
P_{1,i}
\left(
dY_i
-
\sum_j P_{1,j} dY_j
\right)
$$

Where:

$$
\text{diag}(P_1) =
\begin{bmatrix}
p_{11} & 0 & 0 \\
0 & p_{12} & 0 \\
0 & 0 & p_{13}
\end{bmatrix}
$$

$$
P_1^T P_1 =
\begin{bmatrix} p_{11} \\ p_{12} \\ p_{13} \end{bmatrix}
\begin{bmatrix} p_{11} & p_{12} & p_{13} \end{bmatrix}
=
\begin{bmatrix}
p_{11}^2 & p_{11}p_{12} & p_{11}p_{13} \\
p_{11}p_{12} & p_{12}^2 & p_{12}p_{13} \\
p_{11}p_{13} & p_{12}p_{13} & p_{13}^2
\end{bmatrix}
$$

$$
J(P_1) =
\begin{bmatrix} p_{11} & 0 & 0 \\ 0 & p_{12} & 0 \\ 0 & 0 & p_{13} \end{bmatrix}
-
\begin{bmatrix} p_{11}^2 & p_{11}p_{12} & p_{11}p_{13} \\ p_{11}p_{12} & p_{12}^2 & p_{12}p_{13} \\ p_{11}p_{13} & p_{12}p_{13} & p_{13}^2 \end{bmatrix}
=
\begin{bmatrix}
p_{11}(1-p_{11}) & -p_{11}p_{12} & -p_{11}p_{13} \\
-p_{11}p_{12} & p_{12}(1-p_{12}) & -p_{12}p_{13} \\
-p_{11}p_{13} & -p_{12}p_{13} & p_{13}(1-p_{13})
\end{bmatrix}
$$

And this is for one row. This part is to grab the concept right and we will work on writing kernels which will make this procress even transparent.

Now we will be looking at:

$$S = QK^T$$

Here, the upstream gradient backpropagating into this operation will be $\frac{\partial L}{\partial S}$.

The dot product for an individual element $s_{ij}$ is given by:

$$s_{ij} = \sum_{k} q_{ik} K_{jk}$$

Now let's isolate the component $q_{ij}$ to find its gradient. Using the chain rule:

$$\frac{\partial L}{\partial q_{ij}} = \sum_{k} \frac{\partial L}{\partial s_{ik}} \frac{\partial s_{ik}}{\partial q_{ij}}$$

Therefore, the local gradient is:

$$\frac{\partial s_{ik}}{\partial q_{ij}} = K_{jk}$$

Now similar to the <b>O = PV </b>, our final gradients were,

- $dV = P^T G$
- $dP = GV^T$

Given the attention score formula:

$S = \frac{1}{\sqrt{d_k}} Q K^T$

With incoming gradient $G = dS$, the gradients with respect to the input tensors are:

- $dQ = \frac{1}{\sqrt{d_k}} G K$
- $dK = \frac{1}{\sqrt{d_k}} G^T Q$

<b>Where G in this particular case is d_score gradient from our softmax activation back formula.</b>

$dS = \frac{1}{\sqrt{d_k}}  G: (Q dK^T) + K^T dQ$

Where G is the upstream grading using the chain rule of derivative G = d_scores, just my naming convention.

<i>
Let's ingore that constant term for now, it will be difficult for me to type it down. I will multiply later.
</i>

<b>Not something from physics where we could do the derivative where
one thing is constant.</b>

Lets take on component at a time such that we can read them easily.

$$ dS = G: K^T dQ $$

$$dS = G: K^T dQ =\sum_{i,j} G_{ij} (dQ \, K^T)_{ij} \quad \text{--- vi}$$

$$ G: K^T dQ = \sum_{i,j, K} G_{ij} (dQ_{ik}) ( \, K^T)_{kj} $$

$$
\sum_{k,j} \left( \sum_i dQ_{ik} G_{ij} \right) K^T_{kj} \quad \text{--- vii}
$$

By the defination of transpose matrix:

$$K^T_{kj} = K_{jk}$$

So equation seven becomes

$$
\sum_{k,j} \left( \sum_i dQ_{ik} G_{ij} \right) K_{jk}
= G K : dQ = dL_{1}  \quad \text{--- viii}
$$

Equation viii gives one of the compoenent of the derivative now lets continue.


$$ dS = G: (Q dK^T) $$

$$
\sum_{k,j} \left( \sum_i Q_{ik} G_{ij} \right) dK^T_{kj} 
$$

using the def of transpose

$$
dK_{jk} = \sum_i G_{ij} \, Q_{ik} = \sum_i G^T_{ji} \, Q_{ik} = (G^T  Q)_{jk}
$$

$$
dK = G^T Q
$$

Therefore now with the constant multiplied the equation becomes

- $dQ = \frac{1}{\sqrt{d_k}} G K$
- $dK = \frac{1}{\sqrt{d_k}} G^T Q$


<hr />

I will derive O = PV in better way inorder for me to understand better, this is for me to look back at if I forget how does this work and implemnting in code will even make it clear.

$$V = \begin{bmatrix} V_{11} & V_{12} & V_{13} \\ V_{21} & V_{22} & V_{23} \\ V_{31} & V_{32} & V_{33} \end{bmatrix}$$

$$O = \begin{bmatrix} O_{11} & O_{12} & O_{13} \\ O_{21} & O_{22} & O_{23} \\ O_{31} & O_{32} & O_{33} \end{bmatrix}$$

$$K = \begin{bmatrix} K_{11} & K_{12} & K_{13} \\ K_{21} & K_{22} & K_{23} \\ K_{31} & K_{32} & K_{33} \end{bmatrix}$$

O = PV

Here, the upstream gradient backpropagating from the loss function down to this layer is:

$$\frac{\partial L}{\partial O} = \begin{bmatrix} \frac{\partial L}{\partial O_{11}} & \frac{\partial L}{\partial O_{12}} & \frac{\partial L}{\partial O_{13}} \\ \frac{\partial L}{\partial O_{21}} & \frac{\partial L}{\partial O_{22}} & \frac{\partial L}{\partial O_{23}} \\ \frac{\partial L}{\partial O_{31}} & \frac{\partial L}{\partial O_{32}} & \frac{\partial L}{\partial O_{33}} \end{bmatrix}$$

$$
O = \begin{bmatrix} O_{11} & O_{12} & O_{13} \\ O_{21} & O_{22} & O_{23} \\ O_{31} & O_{32} & O_{33} \end{bmatrix} = \begin{bmatrix}
p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31} & p_{11}v_{12} + p_{12}v_{22} + p_{13}v_{32} & p_{11}v_{13} + p_{12}v_{23} + p_{13}v_{33} \\
p_{21}v_{11} + p_{22}v_{21} + p_{23}v_{31} & p_{21}v_{12} + p_{22}v_{22} + p_{23}v_{32} & p_{21}v_{13} + p_{22}v_{23} + p_{23}v_{33} \\
p_{31}v_{11} + p_{32}v_{21} + p_{33}v_{31} & p_{31}v_{12} + p_{32}v_{22} + p_{33}v_{32} & p_{31}v_{13} + p_{32}v_{23} + p_{33}v_{33}
\end{bmatrix}
$$

$$O_{11} = p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31}$$

This opreation happens column wise.

From our matrix equation for $O$, the terms containing $v_{11}$ appear in the first column ($O_{11}, O_{21}, O_{31}$):

$$O_{11} = p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31}$$
$$O_{21} = p_{21}v_{11} + p_{22}v_{21} + p_{23}v_{31}$$
$$O_{31} = p_{31}v_{11} + p_{32}v_{21} + p_{33}v_{31}$$

Applying the multivariate chain rule:

$$\frac{\partial L}{\partial v_{11}} = \frac{\partial L}{\partial O_{11}} \frac{\partial O_{11}}{\partial v_{11}} + \frac{\partial L}{\partial O_{21}} \frac{\partial O_{21}}{\partial v_{11}} + \frac{\partial L}{\partial O_{31}} \frac{\partial O_{31}}{\partial v_{11}}$$

Lets take the expression

$$O_{11} = p_{11}v_{11} + p_{12}v_{21} + p_{13}v_{31}$$

$$\frac{\partial O_{11}}{\partial v_{11}} = p_{11} \frac{\partial v_{11}}{\partial v_{11}} = p_{11}$$

Following the same logic for the other elements in that column:

$$\frac{\partial O_{21}}{\partial v_{11}} = p_{21}$$

$$\frac{\partial O_{31}}{\partial v_{11}} = p_{31}$$

Substituting those local derivatives back into the chain rule expression gives:

$$\frac{\partial L}{\partial v_{11}} = \frac{\partial L}{\partial O_{11}} p_{11} + \frac{\partial L}{\partial O_{21}} p_{21} + \frac{\partial L}{\partial O_{31}} p_{31}$$

So this is equal to the:-

$$\frac{\partial L}{\partial V} = P^T G$$

### Layer norm back propagation formula

Let us consider a function 

$$f(x) = y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$$

Also expressed as,

$$y = \gamma \hat{x} + \beta$$

where

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

where 

$\varepsilon$ = constant  = 1e^-8 


$\gamma$ is a learnable parameter.

$\beta$ is a learnable paramater.

$\mu$ is the mean

$\sigma$ is the standard deviation

For multi variable chain rule we will trace down the depencencies.


$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} = \frac{x_i - \frac{1}{d}\sum_{m=1}^{d}x_m}{\sqrt{\frac{1}{d}\sum_{k=1}^{d}\left(x_k - \frac{1}{d}\sum_{m=1}^{d}x_m\right)^2 + \varepsilon}}$$

where

$$\sigma = \sqrt{\frac{1}{d}\sum_{k=1}^{d}\left(x_k - \frac{1}{d}\sum_{m=1}^{d}x_m\right)^2 + \epsilon}$$

$\sigma$ can also be defined as

$$\sigma = \sqrt{\sigma^2 + \varepsilon} $$

And, 
$$\sigma^2 = \frac{1}{d}\sum_{k=1}^{d}\left(x_k - \frac{1}{d}\sum_{m=1}^{d}x_m\right)^2
=var
$$

$$
u := x - \mu
$$

$$
\hat{x}_i = \frac{u}{\sqrt{\sigma^2 + \varepsilon}}
$$

using the multi variable chain rule we have

<b>Derived at the very end</b>
$$\frac{\partial L}{\partial \hat{x}_i} = G \cdot \gamma$$

Where G is the upstream gradient.

$$
\frac{\partial L}{\partial x_i}
=
\sum_{j=1}^{d}
\frac{\partial L}{\partial \hat{x}_j}
\frac{\partial \hat{x}_j}{\partial x_i}
$$


$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \underbrace{\frac{\partial \hat{x}_j}{\partial u_j}}_{= 1/\sigma} \frac{\partial u_j}{\partial x_i} + \frac{\partial \hat{x}_j}{\partial \sigma} \frac{\partial \sigma}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i} \right]$$


Now lets find the partial derivative of each compoenent above.



<b>First:</b>
$$
\hat{x}_i = \frac{u}{\sqrt{\sigma^2 + \varepsilon}}
$$

$$\frac{\partial \hat{x}_j}{\partial u_j} = \frac{\partial u_j}{\partial u_j} \cdot \frac{1}{\sqrt{\sigma^2 + \varepsilon}} = \frac{1}{\sqrt{\sigma^2 + \varepsilon}} = \frac{1}{\sigma}$$

<hr />

Note: for $\mu$

$$\frac{\partial \mu}{\partial x_i} = \frac{\partial}{\partial x_i} \left( \frac{1}{d} \sum_{k=1}^{d} x_k \right) = \frac{1}{d}$$

where $\delta_{ij}$ is the Kronecker delta defined as:

$$\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$
<hr/>

<b>Second component:</b>

$$\frac{\partial u_j}{\partial x_i} = \frac{\partial (x_j - \mu)}{\partial x_i} = \delta_{ij} - \frac{1}{d}$$

<hr/>

**Third component:** $\frac{\partial \hat{x}}{\partial \sigma}$

$$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}  $$

$$\frac{\partial \hat{x}_i}{\partial \sigma} = \frac{\partial}{\partial \sigma} \left[ (x_i - \mu) (\sigma^2 + \varepsilon)^{-1/2} \right]$$

$$\frac{\partial \hat{x}_i}{\partial \sigma} = (x_i - \mu) \frac{\partial}{\partial \sigma} \left[ (\sigma^2 + \varepsilon)^{-1/2} \right]$$

$$\frac{\partial \hat{x}_i}{\partial \sigma} = (x_i - \mu) \left[ -\frac{1}{2} (\sigma^2 + \varepsilon)^{-3/2} \frac{\partial}{\partial \sigma} (\sigma^2 + \varepsilon) \right]$$

$$\frac{\partial \hat{x}_i}{\partial \sigma} = -\frac{\sigma(x_i - \mu)}{(\sigma^2 + \varepsilon)^{3/2}}$$


<hr />

**Fourth component:** $\frac{\partial \sigma}{\partial \sigma^2}$

Note: 
$$ 

\sigma = \sqrt{\frac{1}{d}\sum_{k=1}^{d}\left(x_k - \frac{1}{d}\sum_{m=1}^{d}x_m\right)^2}

$$


$$\frac{\partial \sigma}{\partial \sigma^2} = \frac{\partial}{\partial \sigma^2} \left[ (\sigma^2)^{1/2} \right] = \frac{1}{2}(\sigma^2)^{-1/2} = \frac{1}{2\sigma}$$

<hr />

**Fifth component:** $\frac{\partial \sigma^2}{\partial x_i}$

$$\frac{\partial \sigma^2}{\partial x_i} = \frac{\partial}{\partial x_i} \left[ \frac{1}{d} \sum_{k=1}^{d} (x_k - \mu)^2 \right]$$

$$\frac{\partial \sigma^2}{\partial x_i} = \frac{1}{d} \sum_{k=1}^{d} 2(x_k - \mu)^{2-1} \cdot \frac{\partial}{\partial x_i}(x_k - \mu)$$

$$\frac{\partial \sigma^2}{\partial x_i} = \frac{2}{d}(x_i - \mu)$$

<hr />

Re-writing the chain rule:

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \underbrace{\frac{\partial \hat{x}_j}{\partial u_j}}_{= 1/\sigma} \frac{\partial u_j}{\partial x_i} + \frac{\partial \hat{x}_j}{\partial \sigma} \frac{\partial \sigma}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i} \right]$$

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \frac{1}{\sigma} \left( \delta_{ij} - \frac{1}{d} \right) + \left( -\frac{\sigma(x_j - \mu)}{(\sigma^2 + \varepsilon)^{3/2}} \right) \left( \frac{1}{2\sigma} \right) \left( \frac{2}{d}(x_i - \mu) \right) \right]$$

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \frac{1}{\sigma} \left( \delta_{ij} - \frac{1}{d} \right) + \left( -\frac{\sigma(x_j - \mu)}{(\sigma^2 + \varepsilon)^{3/2}} \right) \left( \frac{1}{\sigma} \right) \left( \frac{1}{d}(x_i - \mu) \right) \right]$$

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \frac{1}{\sigma} \left( \frac{d\delta_{ij} - 1}{d} \right) - \frac{(x_j - \mu)(x_i - \mu)}{d(\sigma^2 + \varepsilon)^{3/2}} \right]$$

Recall:
 $$
u := x - \mu
$$

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \frac{1}{\sigma} \left( \frac{d\delta_{ij} - 1}{d} \right) - \frac{u_j u_i}{d(\sigma^2 + \varepsilon)^{3/2}} \right]$$

Note:

$$
(\sigma^2 + \varepsilon)^{3/2} = \sigma^3
$$

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ \frac{1}{\sigma} \left( \frac{d\delta_{ij} - 1}{d} \right) - \frac{u_j u_i}{d \sigma^3} \right]$$


$$\frac{\partial L}{\partial x_i} = \frac{1}{d\sigma} \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \left[ (d\delta_{ij} - 1) - \frac{u_j u_i}{\sigma^2} \right]$$

$$\frac{\partial L}{\partial x_i} = \frac{1}{d\sigma} \left[ \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} (d\delta_{ij} - 1) - \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \frac{u_j u_i}{\sigma^2} \right]$$

$$\frac{\partial L}{\partial x_i} = \frac{1}{d\sigma} \left[ d \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} - \frac{u_i}{\sigma} \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \frac{u_j}{\sigma} \right]$$


Now: 

$$\frac{u_i}{\sigma} = \hat{x}_i \quad \text{and} \quad \frac{u_j}{\sigma} = \hat{x}_j$$

$$\frac{\partial L}{\partial x_i} = \frac{1}{d\sigma} \left[ d \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^{d} \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right]$$

Let's keep track on few things


$$y = \gamma \hat{x} + \beta$$

$$\frac{\partial y_i}{\partial \hat{x}_i} = \gamma$$

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial \hat{x}_i}$$

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

$$\frac{\partial L}{\partial \hat{x}_i} = G \cdot \gamma$$

Where G is the upstream gradient.

That G is not something that simple here I just realized we need to account for Linear Transformation that goes in when we have Key, Query and Value and turns out G is:

Recall formula from linear layer,
This is when we have a linear equation:

$$
z = Wh + b
$$

$$
\frac{\partial L}{\partial h} = W^T \frac{\partial L}{\partial z}
$$

$$G_{\hat{x}_0} = \left( \frac{1}{\sqrt{d_k}} G_S K \right) W_Q^T + \left( \frac{1}{\sqrt{d_k}} G_S^T Q \right) W_K^T + (P^T G_O) W_V^T$$

I totally forgot gamma and beta are also learnable paramaters, therefore we will need to do something about them.





Hope that made sense, I may even not remember this after long but when I go through this again should make sense.



### Backpropagation  Derivation and Implementation

**Author:** Avash Lamichhane
**Date:** 2026

### References

- Dao et al., FlashAttention (2022) https://arxiv.org/abs/2205.14135
