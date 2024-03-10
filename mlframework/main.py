


def main():
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    model = NeuralNetwork()
    loss_fn = CrossEntropyLoss()
    optimizer = SGW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}\n-------------------------------')

        loss_sum = 0
        correct_item_count = 0
        item_count = 0
        for batch_index, (x, y) in enumerate(data):
            # x - input; y - true_value
            y_prime = model(x)
            loss = loss_fn(y_prime, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_item_count += (y_prime.argmax(1) == y).sum().item()
            loss_sum += loss.item()
            item_count += len(x)

        average_loss = loss_sum / item_count
        accuracy = correct_item_count / item_count
        print(f'loss: {average_loss:>8f}, ' + f'accuracy: {accuracy * 100:>0.1f}%')


