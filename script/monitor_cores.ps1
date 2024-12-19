while ($true) {

    # Get the load percentage for each CPU core
    $cpuLoad = Get-Counter '\Processor(*)\% Processor Time'

    # Display the load percentage for each core
    $cpuLoad.CounterSamples | ForEach-Object {
        $core = $_.InstanceName
        $load = $_.CookedValue
        Write-Output "Core ${core}: ${load}% load"
        # Get the process information
        $processes = Get-Process | Sort-Object CPU -Descending | Select-Object -First 1

        # Display the top process information
        $processes | ForEach-Object {
            $id = $_.Id
            $name = $_.Name
            $cpu = $_.CPU
            Write-Output "Top Process ID: $id, Name: $name, CPU Time: $cpu"
        }

    }

    
    Start-Sleep -Milliseconds 500
    Clear-Host
}