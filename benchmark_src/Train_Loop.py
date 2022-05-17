def get_accuracy(model, dataset, batch_size, useGPU=True, useAllDataForAccuracy=False):
    # To save time, only evaluate accuracy for 2.5% of the total number of batches at random if useAllDataForAccuracy == False

    # start = time.time()

    if useAllDataForAccuracy == False:
        shuffle = True
    else:
        shuffle = False

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, shuffle=shuffle)

    all_batch_dice = []
    for i, batch in enumerate(data_loader):
        imgs, masks, _ = batch
        imgs = imgs.float()

        # To Enable GPU Usage
        if useGPU and torch.cuda.is_available():
            imgs = imgs.cuda()
            masks = masks.cuda()
            model = model.cuda()

        model = model.eval()
        output = model(imgs)

        batch_dice = dice_channel_batch(output, masks, 0.2).detach().cpu().numpy()
        # print("Batch Dice:", batch_dice)
        all_batch_dice.append(float(batch_dice))

        if useAllDataForAccuracy == False and i >= int(0.025 * len(data_loader)):
            break
    # print("Time in get_accuracy:", time.time() - start)
    return (np.mean(np.asarray(all_batch_dice)))


def plot_training_curve(iters, losses, train_acc, val_acc):
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

    print("Maximum Training Accuracy: {0}. Epoch: {1}".format(max(train_acc), np.argmax(train_acc)))
    print("Maximum Validation Accuracy: {0}. Epoch {1}".format(max(val_acc), np.argmax(val_acc)))


def train(model, train_dataset, val_dataset, batch_size=64, num_epochs=1, learning_rate=0.01, momentum=0.9,
          useGPU=True, saveWeights=True, printIterations=False, useAdams=True, useAllDataForAccuracy=False,
          useDiceLoss=False):

    # Put data in data loaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    num_workers=0, shuffle=False)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                  num_workers=0, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    if useAdams:
        optimizer = optim.Adam(model.parameters(), learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

    iters, losses, train_acc, val_acc = [], [], [], []

    if useGPU and torch.cuda.is_available():
        model = model.cuda()
        print("Training on GPU")

    # training
    n = 0  # the number of iterations
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_data_loader):
            imgs, masks, _ = batch
            imgs = imgs.float()

            # To Enable GPU Usage
            if useGPU and torch.cuda.is_available():
                imgs = imgs.cuda()
                masks = masks.cuda()

            model = model.train()
            out = model(imgs)  # forward pass

            if useDiceLoss:
                loss = calc_loss(out, masks)
            else:
                loss = criterion(out, masks)  # compute the total loss

            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss) / batch_size)  # compute *average* loss

            train_acc.append(get_accuracy(model, train_dataset, batch_size, useGPU,
                                          useAllDataForAccuracy))  # compute training accuracy
            val_acc.append(get_accuracy(model, val_dataset, batch_size, useGPU,
                                        useAllDataForAccuracy))  # compute validation accuracy
            if printIterations:
                print("Iteration: ", str(n), "| Train Loss: ", losses[n], "| Train Accuracy: ", train_acc[n],
                      "| Validation Accuracy: ", val_acc[n])

            n += 1

            # Save the current model (checkpoint) to a file
            if saveWeights:
                model_path = "model_{0}_bs{1}_lr{2}_epoch{3}_iteration{4}".format(model.name,
                                                                                  batch_size,
                                                                                  str(learning_rate).replace('.', '-'),
                                                                                  epoch, n)
                # Save the weights if the validation accuracy is a new maximum
                if val_acc[-1] == max(val_acc):
                    torch.save(model.state_dict(), model_path + ".pth")

                    # Save the very last epoch's weights
    if saveWeights:
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(model.name,
                                                             batch_size,
                                                             str(learning_rate).replace('.', '-'),
                                                             epoch)
        torch.save(model.state_dict(), model_path + ".pth")

    # Write the train/test loss/err into CSV file for plotting later
    if saveWeights:
        epochs = np.arange(1, num_epochs + 1)
        np.savetxt("{}_train_loss.csv".format(model_path), losses)
        np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
        np.savetxt("{}_val_acc.csv".format(model_path), val_acc)

        # plotting
    plot_training_curve(iters, losses, train_acc, val_acc)