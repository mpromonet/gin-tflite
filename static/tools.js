

function drawItem(context, item) {
    context.fillText(item.ClassName, item.Box.Min.X, item.Box.Min.Y);
    context.strokeRect(item.Box.Min.X, item.Box.Min.Y, (item.Box.Max.X - item.Box.Min.X), (item.Box.Max.Y - item.Box.Min.Y));
}


function drawOverlay(context, data) {
    data?.forEach(item => drawItem(context, item));
}