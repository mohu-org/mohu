use mohu_buffer::layout::Order;

#[test]
fn order_default_is_c() {
    assert_eq!(Order::default(), Order::C);
}
