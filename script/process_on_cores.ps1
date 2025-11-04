# Get the load percentage for each CPU core
$cpuLoad = Get-Counter '\Processor(*)\% Processor Time'

# Display the load percentage for each core
$cpuLoad.CounterSamples | ForEach-Object {
    $core = $_.InstanceName
    $load = $_.CookedValue
    Write-Output "Core ${core}: ${load}% load"
}

# Get the process information
$processes = Get-Process | Select-Object Id, Name, CPU

# Display the process information
$processes | ForEach-Object {
    $id = $_.Id
    $name = $_.Name
    $cpu = $_.CPU
    Write-Output "Process ID: $id, Name: $name, CPU Time: $cpu"
}