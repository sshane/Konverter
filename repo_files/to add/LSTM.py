def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, `[batch, num_units]`.
      state: if `state_is_tuple` is False, this must be a state Tensor, `2-D,
        [batch, state_size]`.  If `state_is_tuple` is True, this must be a tuple
        of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    inputs = self._tflite_wrapper.add_input(
        inputs, tag="input", name="input", aggregate="stack", index_override=0)

    # Make sure inputs and bias_initializer has the same type.
    assert inputs.dtype == self.input_to_input_w.dtype

    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    # Note: For TfLite, cell_state is at index 19 while activation state at
    # index 18.
    c_prev = self._tflite_wrapper.add_input(
        c_prev,
        tag="c_prev",
        name="c_prev",
        aggregate="first",
        index_override=19)
    m_prev = self._tflite_wrapper.add_input(
        m_prev,
        tag="m_prev",
        name="m_prev",
        aggregate="first",
        index_override=18)

    input_size = inputs.shape.with_rank(2).dims[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.shape[-1]")

    inputs_and_m_prev = array_ops.concat([inputs, m_prev], axis=1)

    # i stands for input gate.
    # f stands for forget gate activation.
    # o outputs.
    # j output of LSTM unit.
    # c is the final state.
    # m is the output.
    i = nn_ops.bias_add(
        math_ops.matmul(
            inputs_and_m_prev,
            array_ops.concat([self.input_to_input_w, self.cell_to_input_w],
                             axis=1),
            transpose_b=True), self.input_bias)
    f = nn_ops.bias_add(
        math_ops.matmul(
            inputs_and_m_prev,
            array_ops.concat([self.input_to_forget_w, self.cell_to_forget_w],
                             axis=1),
            transpose_b=True), self.forget_bias)
    o = nn_ops.bias_add(
        math_ops.matmul(
            inputs_and_m_prev,
            array_ops.concat([self.input_to_output_w, self.cell_to_output_w],
                             axis=1),
            transpose_b=True), self.output_bias)
    j = nn_ops.bias_add(
        math_ops.matmul(
            inputs_and_m_prev,
            array_ops.concat([self.input_to_cell_w, self.cell_to_cell_w],
                             axis=1),
            transpose_b=True), self.cell_bias)

    # Diagonal connections
    if self._use_peepholes:
      c = (
          sigmoid(f + self._w_f_diag * c_prev) * c_prev +
          sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (sigmoid(f) * c_prev + sigmoid(i) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      transposed_proj_kernel = array_ops.transpose(self._proj_kernel)
      m = math_ops.matmul(m, transposed_proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    c = self._tflite_wrapper.add_output(
        c, tag="c", name="c", aggregate="last", index_override=1)
    m = self._tflite_wrapper.add_output(
        m, tag="m", name="m", index_override=2, aggregate="stack")

    new_state = (
        rnn_cell_impl.LSTMStateTuple(c, m)
        if self._state_is_tuple else array_ops.concat([c, m], 1))
    return m, new_state